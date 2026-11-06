"""Executable neural and knowledge-tracing baselines.

Published names are used only for implementations retaining their defining
mechanism. ``HierarchicalTransformerBaseline`` is intentionally descriptive and
is not presented as a published architecture.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .components import causal_attention_mask, local_session_mask, prior_responses


class CategoricalInput(nn.Module):
    def __init__(self, vocab_sizes: Mapping[str, int], fields: Sequence[str], embedding_dim: int) -> None:
        super().__init__()
        self.fields = tuple(fields)
        self.embeddings = nn.ModuleDict(
            {field: nn.Embedding(vocab_sizes[field], embedding_dim, padding_idx=0) for field in self.fields}
        )

    @property
    def output_dim(self) -> int:
        return sum(embedding.embedding_dim for embedding in self.embeddings.values())

    def forward(self, batch: Mapping[str, Tensor]) -> Tensor:
        return torch.cat([self.embeddings[field](batch[field].long()) for field in self.fields], dim=-1)


class MLPBaseline(nn.Module):
    """Non-sequential embedding MLP; it never consumes the current response."""

    def __init__(self, vocab_sizes: Mapping[str, int], embedding_dim: int = 32, hidden_dim: int = 128,
                 fields: Sequence[str] = ("token", "lemma", "pos", "exercise", "format"), dropout: float = 0.2):
        super().__init__()
        self.input = CategoricalInput(vocab_sizes, fields, embedding_dim)
        self.network = nn.Sequential(
            nn.Linear(self.input.output_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        logits = self.network(self.input(batch)).squeeze(-1)
        return {"logits": logits, "probability": torch.sigmoid(logits)}


class _RecurrentBaseline(nn.Module):
    recurrent_class: type[nn.RNNBase]

    def __init__(self, vocab_sizes: Mapping[str, int], embedding_dim: int = 32, hidden_dim: int = 128,
                 num_layers: int = 1, dropout: float = 0.2) -> None:
        super().__init__()
        self.item = nn.Embedding(vocab_sizes["token"], embedding_dim, padding_idx=0)
        self.response = nn.Embedding(3, embedding_dim, padding_idx=0)
        self.recurrent = self.recurrent_class(
            embedding_dim * 2, hidden_dim, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0, batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        x = torch.cat((self.item(batch["token"].long()), self.response(prior_responses(batch))), -1)
        states, _ = self.recurrent(x)
        logits = self.head(states).squeeze(-1)
        return {"logits": logits, "probability": torch.sigmoid(logits)}


class LSTMBaseline(_RecurrentBaseline):
    recurrent_class = nn.LSTM


class GRUBaseline(_RecurrentBaseline):
    recurrent_class = nn.GRU


class DKT(nn.Module):
    """Original DKT formulation: interaction RNN predicting all next skills."""

    def __init__(self, num_skills: int, embedding_dim: int = 64, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.num_skills = num_skills
        self.interaction = nn.Embedding(num_skills * 2 + 1, embedding_dim, padding_idx=0)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_dim, num_skills)

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        skills = batch["token"].long().clamp(0, self.num_skills - 1)
        prior_skills = torch.zeros_like(skills)
        prior_skills[:, 1:] = skills[:, :-1]
        previous = prior_responses(batch)
        # An unknown previous response (index 0) must map to the padding embedding.
        # Under exercise-level shifting many positions legitimately have no
        # observable previous outcome, and "(0 - 1) * num_skills" would index
        # negatively into the interaction table.
        known = prior_skills.ne(0) & previous.ne(0)
        interaction_id = torch.where(
            known, prior_skills + (previous - 1).clamp_min(0) * self.num_skills, torch.zeros_like(prior_skills)
        )
        state, _ = self.rnn(self.interaction(interaction_id))
        all_logits = self.output(self.dropout(state))
        logits = all_logits.gather(-1, skills.unsqueeze(-1)).squeeze(-1)
        return {"logits": logits, "probability": torch.sigmoid(logits), "all_skill_logits": all_logits}


class DKVMN(nn.Module):
    """Dynamic Key-Value Memory Network with static concept keys and writable values."""

    def __init__(self, num_skills: int, memory_size: int = 50, key_dim: int = 64, value_dim: int = 128):
        super().__init__()
        self.num_skills = num_skills
        self.keys = nn.Parameter(torch.empty(memory_size, key_dim))
        self.initial_values = nn.Parameter(torch.empty(memory_size, value_dim))
        self.question = nn.Embedding(num_skills, key_dim, padding_idx=0)
        self.interaction = nn.Embedding(num_skills * 2 + 1, value_dim, padding_idx=0)
        self.erase = nn.Linear(value_dim, value_dim)
        self.add = nn.Linear(value_dim, value_dim)
        self.summary = nn.Sequential(nn.Linear(key_dim + value_dim, value_dim), nn.Tanh(), nn.Linear(value_dim, 1))
        nn.init.xavier_uniform_(self.keys)
        nn.init.xavier_uniform_(self.initial_values)

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        skills = batch["token"].long()
        responses = batch.get("observed_responses", batch["responses"]).long()
        bsz, length = skills.shape
        values = self.initial_values.unsqueeze(0).expand(bsz, -1, -1).clone()
        # Writing y[t] straight after predicting t would expose it to t+1, which
        # in this task is usually another token of the same exercise whose label
        # is not yet observable. Writes are therefore buffered and committed only
        # once the exercise ends, preserving the canonical update order while
        # matching the information a learner-facing system would actually have.
        exercises = batch["exercise_group"].long() if "exercise_group" in batch else None
        pending: list[tuple[Tensor, Tensor, Tensor]] = []
        logits: list[Tensor] = []

        def commit(mask: Tensor) -> None:
            nonlocal values, pending
            if not bool(mask.any()):
                return
            gate = mask.view(-1, 1, 1).to(values.dtype)
            remaining: list[tuple[Tensor, Tensor, Tensor]] = []
            for correlation, update, done in pending:
                apply_now = mask & ~done
                if bool(apply_now.any()):
                    erase = torch.sigmoid(self.erase(update)).unsqueeze(1)
                    add = torch.tanh(self.add(update)).unsqueeze(1)
                    weight = correlation.unsqueeze(-1)
                    written = values * (1 - weight * erase) + weight * add
                    step_gate = apply_now.view(-1, 1, 1).to(values.dtype)
                    values = step_gate * written + (1 - step_gate) * values
                done = done | mask
                if not bool(done.all()):
                    remaining.append((correlation, update, done))
            pending = remaining
            del gate

        for step in range(length):
            query = self.question(skills[:, step])
            correlation = torch.softmax(query @ self.keys.T, dim=-1)
            read = torch.einsum("bm,bmd->bd", correlation, values)
            logits.append(self.summary(torch.cat((query, read), dim=-1)).squeeze(-1))

            known = responses[:, step] >= 0
            interaction_id = torch.where(
                known,
                skills[:, step] + responses[:, step].clamp_min(0) * self.num_skills,
                torch.zeros_like(skills[:, step]),
            )
            pending.append(
                (correlation, self.interaction(interaction_id), ~known)
            )
            if exercises is None:
                commit(torch.ones(bsz, dtype=torch.bool, device=skills.device))
            elif step + 1 < length:
                commit(exercises[:, step + 1] != exercises[:, step])

        logits_tensor = torch.stack(logits, dim=1)
        return {"logits": logits_tensor, "probability": torch.sigmoid(logits_tensor)}


class SAKT(nn.Module):
    """Self-Attentive Knowledge Tracing with current-skill queries."""

    def __init__(self, num_skills: int, hidden_dim: int = 128, num_heads: int = 4,
                 max_sequence_length: int = 200, dropout: float = 0.2):
        super().__init__()
        self.num_skills = num_skills
        self.skill = nn.Embedding(num_skills, hidden_dim, padding_idx=0)
        self.interaction = nn.Embedding(num_skills * 2 + 1, hidden_dim, padding_idx=0)
        self.position = nn.Embedding(max_sequence_length, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim))
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        skills = batch["token"].long()
        bsz, length = skills.shape
        prior_skills = torch.zeros_like(skills)
        prior_skills[:, 1:] = skills[:, :-1]
        prior_response = prior_responses(batch)
        known = prior_skills.ne(0) & prior_response.ne(0)
        history_ids = torch.where(
            known, prior_skills + (prior_response - 1).clamp_min(0) * self.num_skills, torch.zeros_like(prior_skills)
        )
        pos = self.position(torch.arange(length, device=skills.device)).unsqueeze(0)
        query, history = self.skill(skills) + pos, self.interaction(history_ids) + pos
        attended, weights = self.attention(query, history, history, attn_mask=causal_attention_mask(length, skills.device),
                                           key_padding_mask=batch.get("padding_mask"), need_weights=True)
        state = self.norm1(query + attended)
        state = self.norm2(state + self.ffn(state))
        logits = self.head(state).squeeze(-1)
        return {"logits": logits, "probability": torch.sigmoid(logits), "attention": weights}


class AKT(nn.Module):
    """AKT-style monotonic attention with question and response embeddings.

    This retains AKT's defining learnable distance-decay attention but omits the
    optional Rasch difficulty parameter when no item difficulty is supplied.
    """

    def __init__(self, num_skills: int, hidden_dim: int = 128, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_skills, self.num_heads, self.head_dim = num_skills, num_heads, hidden_dim // num_heads
        self.skill = nn.Embedding(num_skills, hidden_dim, padding_idx=0)
        self.interaction = nn.Embedding(num_skills * 2 + 1, hidden_dim, padding_idx=0)
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        self.gamma = nn.Parameter(torch.zeros(num_heads))
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.ffn = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim * 4), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim * 4, hidden_dim))
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        skills = batch["token"].long()
        bsz, length = skills.shape
        prior_skills = torch.zeros_like(skills)
        prior_skills[:, 1:] = skills[:, :-1]
        prior_response = prior_responses(batch)
        known = prior_skills.ne(0) & prior_response.ne(0)
        ids = torch.where(
            known, prior_skills + (prior_response - 1).clamp_min(0) * self.num_skills, torch.zeros_like(prior_skills)
        )
        query, memory = self.skill(skills), self.interaction(ids)
        q = self.q(query).view(bsz, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k(memory).view(bsz, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v(memory).view(bsz, length, self.num_heads, self.head_dim).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)
        distance = (torch.arange(length, device=skills.device)[:, None] - torch.arange(length, device=skills.device)[None, :]).clamp_min(0).float()
        scores = scores - F.softplus(self.gamma)[None, :, None, None] * distance[None, None]
        mask = causal_attention_mask(length, skills.device)[None, None]
        if "padding_mask" in batch:
            mask = mask | batch["padding_mask"][:, None, None, :]
        attention = scores.masked_fill(mask, torch.finfo(scores.dtype).min).softmax(-1)
        state = self.out((attention @ v).transpose(1, 2).reshape(bsz, length, -1)) + query
        state = state + self.ffn(state)
        logits = self.head(state).squeeze(-1)
        return {"logits": logits, "probability": torch.sigmoid(logits), "attention": attention}


class SAINTPlus(nn.Module):
    """SAINT+-style item encoder and lagged-response decoder with temporal input."""

    def __init__(self, vocab_sizes: Mapping[str, int], hidden_dim: int = 128, num_heads: int = 4,
                 num_layers: int = 2, max_sequence_length: int = 200, dropout: float = 0.2):
        super().__init__()
        self.item = nn.Embedding(vocab_sizes["token"], hidden_dim, padding_idx=0)
        self.exercise = nn.Embedding(vocab_sizes["exercise"], hidden_dim, padding_idx=0)
        self.response = nn.Embedding(3, hidden_dim, padding_idx=0)
        self.time = nn.Linear(1, hidden_dim)
        self.position = nn.Embedding(max_sequence_length, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(hidden_dim, num_heads, hidden_dim * 4, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        length = batch["token"].shape[1]
        causal = causal_attention_mask(length, batch["token"].device)
        pos = self.position(torch.arange(length, device=batch["token"].device)).unsqueeze(0)
        source = self.item(batch["token"].long()) + self.exercise(batch["exercise"].long()) + pos
        memory = self.encoder(source, mask=causal, src_key_padding_mask=batch.get("padding_mask"))
        elapsed = torch.log1p(batch["elapsed_seconds"].float().clamp_min(0)).unsqueeze(-1)
        target = self.response(prior_responses(batch)) + self.time(elapsed) + pos
        state = self.decoder(target, memory, tgt_mask=causal, memory_mask=causal,
                             tgt_key_padding_mask=batch.get("padding_mask"), memory_key_padding_mask=batch.get("padding_mask"))
        logits = self.head(state).squeeze(-1)
        return {"logits": logits, "probability": torch.sigmoid(logits)}


class DeepFM(nn.Module):
    """DeepFM combining first-order, factorization-machine, and deep terms."""

    def __init__(self, vocab_sizes: Mapping[str, int], fields: Sequence[str] = ("token", "lemma", "pos", "exercise", "format"),
                 embedding_dim: int = 16, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.fields = tuple(fields)
        self.linear = nn.ModuleDict({f: nn.Embedding(vocab_sizes[f], 1, padding_idx=0) for f in fields})
        self.factors = nn.ModuleDict({f: nn.Embedding(vocab_sizes[f], embedding_dim, padding_idx=0) for f in fields})
        self.deep = nn.Sequential(nn.Linear(len(fields) * embedding_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        factors = torch.stack([self.factors[f](batch[f].long()) for f in self.fields], dim=-2)
        first = sum(self.linear[f](batch[f].long()).squeeze(-1) for f in self.fields)
        second = 0.5 * ((factors.sum(-2) ** 2 - (factors ** 2).sum(-2)).sum(-1))
        logits = self.bias + first + second + self.deep(factors.flatten(-2)).squeeze(-1)
        return {"logits": logits, "probability": torch.sigmoid(logits)}


class HierarchicalTransformerBaseline(nn.Module):
    """Descriptive two-level causal Transformer baseline without EIP modules."""

    def __init__(self, vocab_sizes: Mapping[str, int], hidden_dim: int = 128, num_heads: int = 4,
                 num_layers: int = 2, max_sequence_length: int = 200, dropout: float = 0.2):
        super().__init__()
        self.item = nn.Embedding(vocab_sizes["token"], hidden_dim, padding_idx=0)
        self.response = nn.Embedding(3, hidden_dim, padding_idx=0)
        self.position = nn.Embedding(max_sequence_length, hidden_dim)
        def layer() -> nn.TransformerEncoderLayer:
            return nn.TransformerEncoderLayer(
                hidden_dim,
                num_heads,
                hidden_dim * 4,
                dropout,
                batch_first=True,
            )
        self.local = nn.TransformerEncoder(layer(), num_layers)
        self.global_encoder = nn.TransformerEncoder(layer(), num_layers)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, batch: Mapping[str, Tensor]) -> dict[str, Tensor]:
        length = batch["token"].shape[1]
        x = self.item(batch["token"].long()) + self.response(prior_responses(batch))
        x = x + self.position(torch.arange(length, device=x.device)).unsqueeze(0)
        local_mask = local_session_mask(batch["session"].long(), batch.get("padding_mask"))
        # A padded query carries session id 0 and every key is then masked, leaving
        # a row of -inf. torch's encoder softmaxes that to NaN, and because a
        # masked key still contributes weight*value with weight 0, the NaN spreads
        # to every valid position in the next encoder. Permitting self-attention
        # keeps each row finite; padded positions are never scored regardless.
        identity = torch.eye(length, dtype=torch.bool, device=x.device)
        local_mask = local_mask & ~identity
        # TransformerEncoder accepts one 3-D mask per attention head.
        heads = self.local.layers[0].self_attn.num_heads
        local_mask = local_mask.repeat_interleave(heads, dim=0)
        # Padding is already encoded in local_mask; passing it again as a key
        # padding mask is what produced the fully-masked rows.
        x = self.local(x, mask=local_mask)
        x = self.global_encoder(x, mask=causal_attention_mask(length, x.device), src_key_padding_mask=batch.get("padding_mask"))
        logits = self.head(x).squeeze(-1)
        return {"logits": logits, "probability": torch.sigmoid(logits)}
