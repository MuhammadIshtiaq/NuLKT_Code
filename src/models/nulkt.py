"""NuLKT: neural uncertainty-aware linguistic knowledge tracing model."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .components import (
    ExerciseContextEncoder,
    CompletedExerciseStateEncoder,
    GatedFeatureFusion,
    exercise_causal_mask,
    LearnerConditionedMoE,
    LearnerItemResidual,
    LinguisticKnowledgeEncoder,
    TemporalTransformerBlock,
    Time2Vec,
    local_session_mask,
    shift_responses,
    shift_responses_by_exercise,
    supervised_contrastive_loss,
)


@dataclass(frozen=True)
class NuLKTConfig:
    """Serializable NuLKT architecture configuration."""

    vocab_sizes: dict[str, int]
    hidden_dim: int = 128
    embedding_dim: int = 64
    num_heads: int = 4
    local_layers: int = 2
    global_layers: int = 2
    num_experts: int = 5
    dropout: float = 0.2
    max_sequence_length: int = 200
    time_embedding_dim: int = 16
    learner_embedding_dim: int = 32
    learner_history_features: int = 6
    # Prompt identity and exercise-level scalars. Off by default so existing
    # checkpoints keep loading and every reported comparison stays like-for-like
    # until the arm is trained and measured on its own terms.
    exercise_context: bool = False
    exercise_scalar_features: int = 4
    irt_residual: bool = False
    completed_exercise_state: bool = False
    contrastive_temperature: float = 0.1
    contrastive_max_samples: int | None = 2048
    uncertainty_threshold: float = 0.45
    # Every token of an exercise is predicted simultaneously and the exercise's
    # surface form is given, so attention within an exercise may be
    # bidirectional. Labels stay hidden via shift_responses_by_exercise.
    within_exercise_bidirectional: bool = False
    ablations: dict[str, bool] = field(default_factory=dict)

    def __post_init__(self) -> None:
        required = {*LinguisticKnowledgeEncoder.DEFAULT_FIELDS, "learner"}
        missing = sorted(required - self.vocab_sizes.keys())
        if missing:
            raise ValueError(f"vocab_sizes missing required fields: {missing}")
        if self.hidden_dim % self.num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.contrastive_max_samples is not None and self.contrastive_max_samples < 2:
            raise ValueError("contrastive_max_samples must be at least 2 or None")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NuLKT(nn.Module):
    """Hierarchical, time-aware, personalized error-prediction architecture.

    Required batch fields are the seven linguistic fields, ``learner``,
    ``responses``, ``timestamps``, ``elapsed_seconds``, and ``session``.
    ``history_features`` and ``padding_mask`` are optional. Current responses are
    shifted internally, preventing the prediction at t from observing y[t].
    """

    def __init__(self, config: NuLKTConfig) -> None:
        super().__init__()
        self.config = config
        c = config
        self.linguistic = LinguisticKnowledgeEncoder(
            c.vocab_sizes, c.embedding_dim, c.hidden_dim, c.dropout
        )
        self.learner_embedding = nn.Embedding(
            c.vocab_sizes["learner"], c.learner_embedding_dim, padding_idx=0
        )
        self.response_embedding = nn.Embedding(3, c.embedding_dim, padding_idx=0)
        self.history_projection = nn.Sequential(
            nn.Linear(c.learner_history_features, c.hidden_dim), nn.GELU(), nn.LayerNorm(c.hidden_dim)
        )
        self.time2vec = Time2Vec(c.time_embedding_dim)
        fusion_dims = [
            c.hidden_dim, c.learner_embedding_dim, c.embedding_dim, c.hidden_dim, c.time_embedding_dim
        ]
        self.exercise_context: ExerciseContextEncoder | None = None
        if c.exercise_context:
            self.exercise_context = ExerciseContextEncoder(
                prompt_vocab=c.vocab_sizes.get("prompt", 1),
                scalar_features=c.exercise_scalar_features,
                embedding_dim=c.embedding_dim,
                output_dim=c.hidden_dim,
                dropout=c.dropout,
                token_vocab=c.vocab_sizes.get("token", 0),
            )
            fusion_dims.append(c.hidden_dim)
        self.completed_exercise_encoder: CompletedExerciseStateEncoder | None = None
        if c.completed_exercise_state:
            self.completed_exercise_encoder = CompletedExerciseStateEncoder(c.hidden_dim, c.dropout)
            fusion_dims.append(c.hidden_dim)
        self.input_fusion = GatedFeatureFusion(fusion_dims, c.hidden_dim, c.dropout)
        self.position = nn.Embedding(c.max_sequence_length, c.hidden_dim)
        self.local_blocks = nn.ModuleList(
            TemporalTransformerBlock(c.hidden_dim, c.num_heads, c.dropout) for _ in range(c.local_layers)
        )
        self.global_blocks = nn.ModuleList(
            TemporalTransformerBlock(c.hidden_dim, c.num_heads, c.dropout) for _ in range(c.global_layers)
        )
        if c.ablations.get("trainable_temporal_decay", False):
            for block in [*self.local_blocks, *self.global_blocks]:
                block.attention.log_decay.requires_grad_(False)
                block.attention.log_power.requires_grad_(False)
        self.learner_condition = nn.Linear(c.learner_embedding_dim + c.hidden_dim, c.hidden_dim)
        self.moe = LearnerConditionedMoE(c.hidden_dim, c.num_experts, c.dropout)
        self.prediction_head = nn.Sequential(nn.LayerNorm(c.hidden_dim), nn.Linear(c.hidden_dim, 1))
        self.mastery_head = nn.Sequential(nn.LayerNorm(c.hidden_dim), nn.Linear(c.hidden_dim, 1))
        self.evidence_head = nn.Sequential(nn.LayerNorm(c.hidden_dim), nn.Linear(c.hidden_dim, 2))
        self.irt_residual: LearnerItemResidual | None = None
        if c.irt_residual:
            self.irt_residual = LearnerItemResidual(
                token_vocab=c.vocab_sizes["token"],
                prompt_vocab=c.vocab_sizes.get("prompt", 1),
                learner_vocab=c.vocab_sizes["learner"],
                format_vocab=c.vocab_sizes["format"],
                history_features=c.learner_history_features,
            )

    def _enabled(self, name: str) -> bool:
        return not self.config.ablations.get(name, False)

    def forward(self, batch: dict[str, Tensor], return_diagnostics: bool = True) -> dict[str, Tensor]:
        responses = batch.get("observed_responses", batch["responses"])
        bsz, length = responses.shape
        if length > self.config.max_sequence_length:
            raise ValueError(
                f"Sequence length {length} exceeds configured maximum {self.config.max_sequence_length}"
            )
        padding = batch.get("padding_mask", torch.zeros_like(responses, dtype=torch.bool)).bool()
        timestamps = batch["timestamps"].float()
        linguistic_batch = batch
        if not self._enabled("context"):
            linguistic_batch = dict(batch)
            linguistic_batch["context"] = torch.zeros_like(batch["context"])
        linguistic, linguistic_gates = self.linguistic(linguistic_batch)
        if not self._enabled("linguistic"):
            linguistic = torch.zeros_like(linguistic)

        learner_raw = self.learner_embedding(batch["learner"].long())
        if not self._enabled("personalization"):
            learner_raw = torch.zeros_like(learner_raw)
        if self.config.within_exercise_bidirectional and "exercise_group" in batch:
            previous_responses = self.response_embedding(
                shift_responses_by_exercise(responses, batch["exercise_group"].long())
            )
        else:
            previous_responses = self.response_embedding(shift_responses(responses))
        history = batch.get(
            "history_features",
            torch.zeros(bsz, length, self.config.learner_history_features, device=responses.device),
        ).float()
        history_state = self.history_projection(history)
        temporal = self.time2vec(batch["elapsed_seconds"])
        if not self._enabled("temporal"):
            temporal = torch.zeros_like(temporal)

        branches = [linguistic, learner_raw, previous_responses, history_state, temporal]
        if self.exercise_context is not None:
            context = self.exercise_context(
                batch.get("prompt", torch.zeros_like(responses)),
                batch.get("exercise_scalars"),
                batch.get("neighbour_tokens"),
            )
            if not self._enabled("exercise_context"):
                context = torch.zeros_like(context)
            branches.append(context)
        if self.completed_exercise_encoder is not None:
            completed_state = self.completed_exercise_encoder(
                batch.get(
                    "completed_exercise_state",
                    torch.zeros(bsz, length, 4, device=responses.device),
                )
            )
            branches.append(completed_state)
        fused, input_gates = self.input_fusion(branches)
        positions = torch.arange(length, device=responses.device)
        x = fused + self.position(positions).unsqueeze(0)

        local_attentions: list[Tensor] = []
        if self.config.within_exercise_bidirectional and "exercise_group" in batch:
            # Causal across exercises, unrestricted within the current one.
            session_mask = exercise_causal_mask(batch["exercise_group"].long(), padding)
        else:
            session_mask = local_session_mask(batch["session"].long(), padding)
        bidirectional = self.config.within_exercise_bidirectional and "exercise_group" in batch
        for block in self.local_blocks:
            x, attention = block(x, timestamps, padding, session_mask, causal=not bidirectional)
            local_attentions.append(attention)

        # Global blocks attend causally across local session-aware states. Thus a
        # position sees its current-session prefix and completed prior sessions,
        # but never a later interaction from either level.
        global_attentions: list[Tensor] = []
        if self._enabled("hierarchical"):
            global_mask = (
                exercise_causal_mask(batch["exercise_group"].long(), padding) if bidirectional else None
            )
            for block in self.global_blocks:
                x, attention = block(x, timestamps, padding, global_mask, causal=not bidirectional)
                global_attentions.append(attention)

        learner_state = torch.tanh(self.learner_condition(torch.cat((learner_raw, history_state), dim=-1)))
        if self._enabled("moe"):
            x, routing = self.moe(x, learner_state)
        else:
            routing = {
                "expert_weights": torch.full(
                    (*x.shape[:2], self.config.num_experts),
                    1 / self.config.num_experts,
                    device=x.device,
                ),
                "routing_entropy": torch.full(
                    x.shape[:2], float(torch.log(torch.tensor(self.config.num_experts))), device=x.device
                ),
                "expert_utilization": torch.full(
                    (self.config.num_experts,), 1 / self.config.num_experts, device=x.device
                ),
            }

        logits = self.prediction_head(x).squeeze(-1)
        if self.irt_residual is not None:
            logits = logits + self.irt_residual(batch, history)
        evidence = F.softplus(self.evidence_head(x)) + 1.0
        alpha, beta = evidence.unbind(dim=-1)
        probability = alpha / (alpha + beta)
        uncertainty = 2.0 / (alpha + beta)
        confidence = 1.0 - uncertainty
        elapsed_hours = batch["elapsed_seconds"].float().clamp_min(0) / 3600
        temporal_rate = F.softplus(self.global_blocks[0].attention.log_decay).mean() if self.global_blocks else x.new_tensor(0.1)
        forgetting_signal = 1 - torch.exp(-temporal_rate * torch.log1p(elapsed_hours))
        output = {
            "logits": logits,
            "probability": probability,
            "mastery": torch.sigmoid(self.mastery_head(x).squeeze(-1)),
            "alpha": alpha,
            "beta": beta,
            "uncertainty": uncertainty,
            "confidence": confidence,
            "abstention": uncertainty >= self.config.uncertainty_threshold,
            "forgetting_signal": forgetting_signal,
            "representation": x,
            **routing,
        }
        if return_diagnostics:
            output.update(
                {
                    "linguistic_gates": linguistic_gates,
                    "input_gates": input_gates,
                    "local_attention": torch.stack(local_attentions, dim=1),
                    "global_attention": (
                        torch.stack(global_attentions, dim=1)
                        if global_attentions
                        else x.new_empty(bsz, 0, self.config.num_heads, length, length)
                    ),
                }
            )
        return output

    def contrastive_loss(self, outputs: dict[str, Tensor], batch: dict[str, Tensor]) -> Tensor:
        if not self._enabled("contrastive"):
            return outputs["representation"].sum() * 0
        valid = ~batch.get("padding_mask", torch.zeros_like(batch["responses"], dtype=torch.bool))
        if "score_mask" in batch:
            valid = valid & batch["score_mask"].bool()
        return supervised_contrastive_loss(
            outputs["representation"],
            batch["learner"],
            valid,
            self.config.contrastive_temperature,
            self.config.contrastive_max_samples,
        )
