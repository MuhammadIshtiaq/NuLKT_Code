"""Reusable, leakage-safe modules for knowledge tracing models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def shift_responses(responses: Tensor, padding_value: int = 0) -> Tensor:
    """Right-shift responses so position *t* only observes responses before *t*.

    Response values are expected to be 0/1. They are offset by one so index zero
    remains a dedicated BOS/padding value and correctness maps to indices 1/2.
    """

    if responses.ndim != 2:
        raise ValueError("responses must have shape [batch, sequence]")
    shifted = torch.full_like(responses, padding_value)
    shifted[:, 1:] = responses[:, :-1].long() + 1
    return shifted


def shift_responses_by_exercise(responses: Tensor, exercise_ids: Tensor, padding_value: int = 0) -> Tensor:
    """Expose only responses from strictly earlier exercises.

    In this task every token of an exercise is predicted at once, so no label
    within the current exercise is observable when predicting any of them.
    Token-level shifting alone would let the second token of an exercise consume
    the first token's outcome, which the evaluation setup never provides.
    """

    if responses.shape != exercise_ids.shape:
        raise ValueError("responses and exercise_ids must have the same shape")
    shifted = torch.full_like(responses, padding_value)
    shifted[:, 1:] = responses[:, :-1].long() + 1
    # A position may carry the preceding token's outcome only when that token
    # belongs to an earlier exercise. Later tokens of an exercise still reach
    # that evidence through within-exercise attention, via the first token.
    same_exercise = torch.zeros_like(exercise_ids, dtype=torch.bool)
    same_exercise[:, 1:] = exercise_ids[:, 1:] == exercise_ids[:, :-1]
    return shifted.masked_fill(same_exercise, padding_value)


def prior_responses(batch: Mapping[str, Tensor]) -> Tensor:
    """Observable previous outcomes for every model, under one shared rule.

    Baselines and the proposed model must face the same information constraint or
    a comparison between them measures the constraint rather than the
    architecture. When the batch identifies exercises, no label from the current
    exercise is observable - matching the task, where every token of an exercise
    is predicted at once. Batches without that field fall back to token-level
    shifting so older callers keep working.
    """

    # `responses` remain the scoring targets.  Held-out labels must never be
    # model inputs; their observable history is supplied separately and uses
    # -1 for outcomes unavailable at prediction time.  The shift turns -1 into
    # embedding index 0 (unknown/padding).
    responses = batch.get("observed_responses", batch["responses"])
    if "exercise_group" in batch:
        return shift_responses_by_exercise(responses, batch["exercise_group"].long())
    return shift_responses(responses)


def exercise_causal_mask(exercise_ids: Tensor, padding_mask: Tensor | None = None) -> Tensor:
    """Full attention inside the current exercise, causal across exercises.

    Returns ``[batch, sequence, sequence]`` with True denoting forbidden edges.
    Attending to a later token of the *same* exercise is legitimate because the
    exercise's surface form is given at prediction time; only its labels are
    withheld, and :func:`shift_responses_by_exercise` withholds them.
    """

    if exercise_ids.ndim != 2:
        raise ValueError("exercise_ids must have shape [batch, sequence]")
    later_exercise = exercise_ids.unsqueeze(1) < exercise_ids.unsqueeze(2)
    mask = later_exercise.transpose(1, 2)
    if padding_mask is not None:
        mask = mask | padding_mask.unsqueeze(1)
    return mask


def causal_attention_mask(length: int, device: torch.device | None = None) -> Tensor:
    """Return a boolean mask where True entries are forbidden attention edges."""

    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)


def local_session_mask(session_ids: Tensor, padding_mask: Tensor | None = None) -> Tensor:
    """Build per-example causal masks restricted to the current session.

    Returns ``[batch, sequence, sequence]`` with True denoting masked edges.
    """

    if session_ids.ndim != 2:
        raise ValueError("session_ids must have shape [batch, sequence]")
    bsz, length = session_ids.shape
    future = causal_attention_mask(length, session_ids.device).unsqueeze(0).expand(bsz, -1, -1)
    different_session = session_ids.unsqueeze(2) != session_ids.unsqueeze(1)
    mask = future | different_session
    if padding_mask is not None:
        mask = mask | padding_mask.unsqueeze(1)
    return mask


class Time2Vec(nn.Module):
    """Linear plus periodic representation of log-scaled elapsed time."""

    def __init__(self, output_dim: int) -> None:
        super().__init__()
        if output_dim < 2:
            raise ValueError("Time2Vec output_dim must be at least 2")
        self.linear = nn.Linear(1, 1)
        self.periodic = nn.Linear(1, output_dim - 1)

    def forward(self, elapsed_seconds: Tensor) -> Tensor:
        x = torch.log1p(elapsed_seconds.float().clamp_min(0)).unsqueeze(-1)
        return torch.cat((self.linear(x), torch.sin(self.periodic(x))), dim=-1)


class GatedFeatureFusion(nn.Module):
    """Fuse heterogeneous feature groups through learned normalized gates."""

    def __init__(self, input_dims: Sequence[int], output_dim: int, dropout: float) -> None:
        super().__init__()
        if not input_dims:
            raise ValueError("At least one feature group is required")
        self.projections = nn.ModuleList(nn.Linear(dim, output_dim) for dim in input_dims)
        self.gates = nn.ModuleList(nn.Linear(dim, 1) for dim in input_dims)
        self.residual = nn.Linear(sum(input_dims), output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
        if len(features) != len(self.projections):
            raise ValueError(f"Expected {len(self.projections)} feature groups, got {len(features)}")
        projected = torch.stack([torch.tanh(p(x)) for p, x in zip(self.projections, features)], dim=-2)
        gate_logits = torch.cat([g(x) for g, x in zip(self.gates, features)], dim=-1)
        weights = gate_logits.softmax(dim=-1)
        fused = (projected * weights.unsqueeze(-1)).sum(dim=-2)
        residual = self.residual(torch.cat(list(features), dim=-1))
        return self.norm(residual + self.dropout(fused)), weights


class LinguisticKnowledgeEncoder(nn.Module):
    """Encode lexical, morphological, and exercise information without labels."""

    DEFAULT_FIELDS = ("token", "lemma", "pos", "morph", "context", "exercise", "format")

    def __init__(
        self,
        vocab_sizes: Mapping[str, int],
        embedding_dim: int,
        hidden_dim: int,
        dropout: float,
        fields: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        self.fields = tuple(fields or self.DEFAULT_FIELDS)
        missing = [name for name in self.fields if name not in vocab_sizes]
        if missing:
            raise ValueError(f"Missing vocabulary sizes: {missing}")
        self.embeddings = nn.ModuleDict(
            {name: nn.Embedding(vocab_sizes[name], embedding_dim, padding_idx=0) for name in self.fields}
        )
        self.fusion = GatedFeatureFusion([embedding_dim] * len(self.fields), hidden_dim, dropout)

    def forward(self, batch: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
        missing = [name for name in self.fields if name not in batch]
        if missing:
            raise KeyError(f"Batch lacks linguistic fields: {missing}")
        return self.fusion([self.embeddings[name](batch[name].long()) for name in self.fields])


class TemporalCausalSelfAttention(nn.Module):
    """Causal attention with trainable exponential and power-law recency bias."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.log_decay = nn.Parameter(torch.full((num_heads,), -2.0))
        self.log_power = nn.Parameter(torch.zeros(num_heads))

    def forward(
        self,
        x: Tensor,
        timestamps: Tensor,
        padding_mask: Tensor | None = None,
        extra_mask: Tensor | None = None,
        causal: bool = True,
    ) -> tuple[Tensor, Tensor]:
        bsz, length, dim = x.shape
        qkv = self.qkv(x).view(bsz, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        scores = torch.einsum("bthd,bshd->bhts", q, k) * self.scale
        delta = (timestamps.unsqueeze(2) - timestamps.unsqueeze(1)).clamp_min(0).float()
        delta_hours = delta / 3600.0
        decay = F.softplus(self.log_decay).view(1, -1, 1, 1)
        power = F.softplus(self.log_power).view(1, -1, 1, 1)
        scores = scores - decay * torch.log1p(delta_hours).unsqueeze(1).pow(power)
        # ``extra_mask`` can only forbid additional edges, so a caller that needs
        # bidirectional attention inside a group must also switch the token-level
        # causal mask off and supply a mask that enforces ordering itself.
        mask = (
            causal_attention_mask(length, x.device).view(1, 1, length, length)
            if causal
            else torch.zeros(1, 1, length, length, dtype=torch.bool, device=x.device)
        )
        if padding_mask is not None:
            mask = mask | padding_mask[:, None, None, :]
        if extra_mask is not None:
            mask = mask | extra_mask[:, None, :, :]
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        weights = self.dropout(scores.softmax(dim=-1))
        attended = torch.einsum("bhts,bshd->bthd", weights, v).reshape(bsz, length, dim)
        return self.output(attended), weights


class TemporalTransformerBlock(nn.Module):
    """Pre-normalized Transformer block using temporal causal attention."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, expansion: int = 4) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attention = TemporalCausalSelfAttention(hidden_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * expansion, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: Tensor,
        timestamps: Tensor,
        padding_mask: Tensor | None = None,
        extra_mask: Tensor | None = None,
        causal: bool = True,
    ) -> tuple[Tensor, Tensor]:
        attended, weights = self.attention(self.norm1(x), timestamps, padding_mask, extra_mask, causal)
        x = x + attended
        return x + self.ffn(self.norm2(x)), weights


class LearnerConditionedMoE(nn.Module):
    """Small dense MoE with learner-conditioned routing and utilization stats."""

    def __init__(self, hidden_dim: int, num_experts: int, dropout: float) -> None:
        super().__init__()
        if num_experts < 2:
            raise ValueError("num_experts must be at least two")
        self.experts = nn.ModuleList(
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            for _ in range(num_experts)
        )
        self.router = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, num_experts))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, representation: Tensor, learner_state: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        weights = self.router(torch.cat((representation, learner_state), dim=-1)).softmax(dim=-1)
        expert_outputs = torch.stack([expert(representation) for expert in self.experts], dim=-2)
        output = self.norm(representation + (weights.unsqueeze(-1) * expert_outputs).sum(dim=-2))
        entropy = -(weights.clamp_min(1e-9).log() * weights).sum(dim=-1)
        return output, {
            "expert_weights": weights,
            "routing_entropy": entropy,
            "expert_utilization": weights.mean(dim=(0, 1)),
        }


class LearnerItemResidual(nn.Module):
    """Low-variance Rasch-style residual from learner/item feature evidence.

    Zero-initialized bias tables make enabling the module exactly reproduce the
    base model at initialization, while a unit scale gives every table a useful
    first-step gradient. Index zero is padding/unknown and remains zero.
    """

    def __init__(
        self,
        token_vocab: int,
        prompt_vocab: int,
        learner_vocab: int,
        format_vocab: int,
        history_features: int,
    ) -> None:
        super().__init__()
        sizes = (token_vocab, prompt_vocab, learner_vocab, format_vocab)
        if min(sizes) < 1:
            raise ValueError("Residual vocabularies must be positive")
        self.token_difficulty = nn.Embedding(token_vocab, 1, padding_idx=0)
        self.prompt_difficulty = nn.Embedding(prompt_vocab, 1, padding_idx=0)
        self.learner_ability = nn.Embedding(learner_vocab, 1, padding_idx=0)
        self.format_bias = nn.Embedding(format_vocab, 1, padding_idx=0)
        self.history = nn.Linear(history_features, 1, bias=False)
        self.scale = nn.Parameter(torch.ones(()))
        for embedding in (
            self.token_difficulty,
            self.prompt_difficulty,
            self.learner_ability,
            self.format_bias,
        ):
            nn.init.zeros_(embedding.weight)
            with torch.no_grad():
                embedding.weight[0].zero_()
        nn.init.zeros_(self.history.weight)

    def forward(self, batch: Mapping[str, Tensor], history: Tensor) -> Tensor:
        residual = (
            self.token_difficulty(batch["token"].long())
            + self.prompt_difficulty(batch.get("prompt", torch.zeros_like(batch["token"])).long())
            - self.learner_ability(batch["learner"].long())
            + self.format_bias(batch["format"].long())
            + self.history(history.float())
        ).squeeze(-1)
        return self.scale * residual


class CompletedExerciseStateEncoder(nn.Module):
    """Project four strictly past-only completed-exercise summaries."""

    def __init__(self, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 32),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, state: Tensor) -> Tensor:
        if state.shape[-1] != 4:
            raise ValueError(f"completed exercise state must have width 4, got {state.shape[-1]}")
        return self.network(state.float())


def supervised_contrastive_loss(
    representations: Tensor,
    learner_ids: Tensor,
    valid_mask: Tensor | None = None,
    temperature: float = 0.1,
    max_samples: int | None = 2048,
) -> Tensor:
    """Prefix-level contrastive loss using two views of each causal state.

    Positives are stochastic views of the *same prefix representation*. Later
    prefixes are never designated as positives for earlier ones, avoiding the
    subtle future-response leakage caused by naive same-learner pairing.
    ``learner_ids`` is retained in the API for provenance checks and shape
    validation; negatives may include other prefixes of the same learner.
    """

    reps = representations.reshape(-1, representations.shape[-1])
    ids = learner_ids.reshape(-1)
    valid = torch.ones_like(ids, dtype=torch.bool) if valid_mask is None else valid_mask.reshape(-1).bool()
    reps, ids = reps[valid], ids[valid]
    if reps.shape[0] < 2:
        return representations.sum() * 0
    if ids.shape[0] != reps.shape[0]:
        raise ValueError("learner_ids shape does not match representations")
    if max_samples is not None:
        if max_samples < 2:
            raise ValueError("max_samples must be at least 2 or None")
        if reps.shape[0] > max_samples:
            # Uniformly sample causal prefixes before forming the O(n^2)
            # similarity matrix. This preserves the same-prefix positive pair
            # definition while bounding accelerator memory under the seed.
            selected = torch.randperm(reps.shape[0], device=reps.device)[:max_samples]
            reps, ids = reps[selected], ids[selected]
    view_a = F.normalize(F.dropout(reps, p=0.1, training=True), dim=-1)
    view_b = F.normalize(F.dropout(reps, p=0.1, training=True), dim=-1)
    logits = view_a @ view_b.T / temperature
    labels = torch.arange(reps.shape[0], device=reps.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def beta_evidential_loss(alpha: Tensor, beta: Tensor, targets: Tensor, mask: Tensor | None = None) -> Tensor:
    """Expected Bernoulli NLL under a Beta distribution plus mild evidence penalty."""

    targets = targets.float()
    nll = -(targets * (torch.digamma(alpha) - torch.digamma(alpha + beta)) +
            (1 - targets) * (torch.digamma(beta) - torch.digamma(alpha + beta)))
    incorrect_evidence = targets * beta + (1 - targets) * alpha
    loss = nll + 1e-3 * incorrect_evidence
    if mask is not None:
        return (loss * mask).sum() / mask.sum().clamp_min(1)
    return loss.mean()


class ExerciseContextEncoder(nn.Module):
    """Encode prompt identity and exercise-level scalars given at prediction time.

    Motivated by a measured gap rather than by adding capacity for its own sake.
    Gradient boosting on the official partition draws 32.6% of its total split
    gain from signals the sequence model never receives, and the single largest
    of those is ``prompt`` at 16.2% - the identity of the sentence being
    exercised. Prompt identity is legitimately available at prediction time: the
    SLAM task presents the exercise and withholds only its labels.

    The prompt vocabulary is small enough to embed directly (roughly 5.4k
    distinct prompts over 3.4M interactions), so this is an embedding table
    rather than a text encoder.

    ``scalars`` carries exercise metadata that is likewise given with the
    exercise - response latency, position within the session, token position.
    Latency arrives raw and dirty (negative values and six-figure outliers both
    occur), so it is clipped and log-compressed by the caller, and a
    missing-value indicator travels alongside it rather than a silent zero fill.
    """

    def __init__(
        self,
        prompt_vocab: int,
        scalar_features: int,
        embedding_dim: int,
        output_dim: int,
        dropout: float,
        token_vocab: int = 0,
    ) -> None:
        super().__init__()
        if prompt_vocab < 1:
            raise ValueError("prompt_vocab must be at least 1")
        if scalar_features < 0:
            raise ValueError("scalar_features cannot be negative")
        self.prompt_embedding = nn.Embedding(prompt_vocab, embedding_dim, padding_idx=0)
        self.scalar_features = scalar_features
        width = embedding_dim
        if scalar_features:
            self.scalar_projection = nn.Sequential(
                nn.Linear(scalar_features, embedding_dim), nn.GELU(), nn.LayerNorm(embedding_dim)
            )
            width += embedding_dim
        # Neighbouring-token identity shares the target token's vocabulary, but
        # deliberately not its embedding table: "this word, as the target" and
        # "this word, as the word before" carry different information, and tying
        # them would force one vector to mean both.
        self.neighbour_embedding: nn.Embedding | None = None
        if token_vocab > 0:
            self.neighbour_embedding = nn.Embedding(token_vocab, embedding_dim // 2, padding_idx=0)
            width += embedding_dim  # two neighbours at half width each
        self.output = nn.Sequential(
            nn.Linear(width, output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self, prompt: Tensor, scalars: Tensor | None = None, neighbours: Tensor | None = None
    ) -> Tensor:
        encoded = self.prompt_embedding(prompt.long())
        if self.scalar_features:
            if scalars is None:
                scalars = encoded.new_zeros((*prompt.shape, self.scalar_features))
            encoded = torch.cat((encoded, self.scalar_projection(scalars.float())), dim=-1)
        if self.neighbour_embedding is not None:
            if neighbours is None:
                neighbours = torch.zeros((*prompt.shape, 2), dtype=torch.long, device=prompt.device)
            embedded = self.neighbour_embedding(neighbours.long())
            encoded = torch.cat((encoded, embedded.flatten(start_dim=-2)), dim=-1)
        return self.output(encoded)
