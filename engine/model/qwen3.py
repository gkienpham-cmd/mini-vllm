"""Readable Qwen3 dense-model forward pass implemented in plain PyTorch."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from engine.cache import AppendReservation, PagedKVCache, SequenceId, paged_attention
from engine.config import EngineConfig


class Qwen3RMSNorm(nn.Module):
    """RMSNorm with FP32 reduction for stable CPU and FP16 behavior."""

    def __init__(self, size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        # Squaring FP16 activations can overflow or lose small variance terms.
        states_fp32 = hidden_states.to(torch.float32)
        variance = states_fp32.square().mean(dim=-1, keepdim=True)
        normalized = states_fp32 * torch.rsqrt(variance + self.eps)
        return self.weight * normalized.to(input_dtype)


def rotate_half(hidden_states: torch.Tensor) -> torch.Tensor:
    """Match Hugging Face's split-half Qwen3 RoPE convention."""

    first_half, second_half = hidden_states.chunk(2, dim=-1)
    return torch.cat((-second_half, first_half), dim=-1)


class Qwen3RotaryEmbedding(nn.Module):
    """Fixed-context RoPE tables precomputed once in FP32."""

    def __init__(self, config: EngineConfig) -> None:
        super().__init__()
        frequency_indices = torch.arange(0, config.head_dim, 2, dtype=torch.float32)
        inverse_frequency = 1.0 / (
            config.rope_theta ** (frequency_indices / config.head_dim)
        )
        positions = torch.arange(config.max_position_embeddings, dtype=torch.float32)
        frequencies = torch.outer(positions, inverse_frequency)

        # Qwen3 duplicates frequency halves; it does not interleave even/odd pairs.
        full_frequencies = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("cosine", full_frequencies.cos(), persistent=False)
        self.register_buffer("sine", full_frequencies.sin(), persistent=False)

    def forward(
        self,
        position_ids: torch.Tensor,
        *,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if position_ids.ndim != 2:
            raise ValueError("position_ids must have shape [batch, sequence]")
        if position_ids.numel() and int(position_ids.max()) >= self.cosine.shape[0]:
            raise ValueError("position_ids exceed the precomputed RoPE context")

        # The singleton head axis broadcasts one table across Q and KV heads.
        cosine = self.cosine[position_ids].unsqueeze(1).to(dtype=dtype)
        sine = self.sine[position_ids].unsqueeze(1).to(dtype=dtype)
        return cosine, sine


def apply_rotary_embedding(
    query: torch.Tensor,
    key: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        query * cosine + rotate_half(query) * sine,
        key * cosine + rotate_half(key) * sine,
    )


def repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    """Map each KV head to its adjacent group of query heads without copies."""

    if repeats == 1:
        return hidden_states
    batch, kv_heads, sequence, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(
        batch, kv_heads, repeats, sequence, head_dim
    )
    return expanded.reshape(batch, kv_heads * repeats, sequence, head_dim)


class Qwen3Attention(nn.Module):
    def __init__(self, config: EngineConfig, rotary: Qwen3RotaryEmbedding) -> None:
        super().__init__()
        self.config = config
        self.rotary = rotary
        self.scaling = config.head_dim**-0.5

        self.q_proj = nn.Linear(
            config.hidden_size,
            config.query_projection_size,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.key_value_projection_size,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.key_value_projection_size,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.query_projection_size,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(config.head_dim, config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        query, key, value = self._project_qkv(hidden_states, position_ids)

        key = repeat_kv(key, self.config.query_heads_per_kv_head)
        value = repeat_kv(value, self.config.query_heads_per_kv_head)

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scaling
        scores = scores + attention_mask
        # FP32 softmax keeps masked, low-probability values stable in FP16.
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        attended = torch.matmul(probabilities, value)

        return self._project_output(attended)

    def forward_cached(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        cache: PagedKVCache,
        reservation: AppendReservation,
        layer_index: int,
    ) -> torch.Tensor:
        """Append K/V, gather through block tables, and attend to cached context."""

        query, key, value = self._project_qkv(hidden_states, position_ids)
        cache.write(layer_index, reservation, key, value)
        layer_cache = cache.layers[layer_index]
        attended = paged_attention(
            query,
            key_blocks=layer_cache.key,
            value_blocks=layer_cache.value,
            block_tables=reservation.block_tables,
            context_lengths=reservation.context_lengths,
            query_start_positions=reservation.query_start_positions,
            block_size=self.config.kv_block_size,
            query_heads_per_kv_head=self.config.query_heads_per_kv_head,
            scaling=self.scaling,
        )
        return self._project_output(attended)

    def _project_qkv(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, sequence, _ = hidden_states.shape

        # Normalize after projection so each head gets its own head_dim-value RMS.
        query = self.q_proj(hidden_states).view(
            batch, sequence, self.config.num_attention_heads, self.config.head_dim
        )
        key = self.k_proj(hidden_states).view(
            batch, sequence, self.config.num_key_value_heads, self.config.head_dim
        )
        value = self.v_proj(hidden_states).view(
            batch, sequence, self.config.num_key_value_heads, self.config.head_dim
        )
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)

        cosine, sine = self.rotary(position_ids, dtype=query.dtype)
        query, key = apply_rotary_embedding(query, key, cosine, sine)
        return query, key, value

    def _project_output(self, attended: torch.Tensor) -> torch.Tensor:
        batch, _, sequence, _ = attended.shape
        # Head order must be restored before the output projection.
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(batch, sequence, self.config.query_projection_size)
        return self.o_proj(attended)


class Qwen3MLP(nn.Module):
    def __init__(self, config: EngineConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gated = F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return self.down_proj(gated)


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: EngineConfig, rotary: Qwen3RotaryEmbedding) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(config, rotary)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

    def forward_cached(
        self,
        hidden_states: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        cache: PagedKVCache,
        reservation: AppendReservation,
        layer_index: int,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn.forward_cached(
            hidden_states,
            position_ids=position_ids,
            cache=cache,
            reservation=reservation,
            layer_index=layer_index,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3Model(nn.Module):
    def __init__(self, config: EngineConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer(config, self.rotary_emb)
            for _ in range(config.num_hidden_layers)
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        batch, sequence = input_ids.shape
        if sequence == 0:
            raise ValueError("input_ids cannot be empty")
        if sequence > self.config.max_position_embeddings:
            raise ValueError("input sequence exceeds the configured context length")

        if position_ids is None:
            position_ids = torch.arange(sequence, device=input_ids.device)[None, :]
            position_ids = position_ids.expand(batch, -1)
        elif position_ids.shape != input_ids.shape:
            raise ValueError("position_ids must match input_ids shape")

        hidden_states = self.embed_tokens(input_ids)
        additive_mask = self._build_causal_mask(
            input_ids,
            attention_mask=attention_mask,
            dtype=hidden_states.dtype,
        )

        captured: list[torch.Tensor] | None = [] if output_hidden_states else None
        for layer in self.layers:
            if captured is not None:
                # Match HF: capture the input to each decoder layer.
                captured.append(hidden_states)
            hidden_states = layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=additive_mask,
            )

        hidden_states = self.norm(hidden_states)
        if captured is not None:
            captured.append(hidden_states)
        return hidden_states, tuple(captured) if captured is not None else None

    def forward_cached(
        self,
        input_ids: torch.Tensor,
        *,
        position_ids: torch.Tensor,
        cache: PagedKVCache,
        reservation: AppendReservation,
        output_hidden_states: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...] | None]:
        """Run the explicit paged path for a uniform-length append batch."""

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.shape != position_ids.shape:
            raise ValueError("position_ids must match input_ids shape")

        hidden_states = self.embed_tokens(input_ids)
        captured: list[torch.Tensor] | None = [] if output_hidden_states else None
        for layer_index, layer in enumerate(self.layers):
            if captured is not None:
                captured.append(hidden_states)
            hidden_states = layer.forward_cached(
                hidden_states,
                position_ids=position_ids,
                cache=cache,
                reservation=reservation,
                layer_index=layer_index,
            )

        hidden_states = self.norm(hidden_states)
        if captured is not None:
            captured.append(hidden_states)
        return hidden_states, tuple(captured) if captured is not None else None

    @staticmethod
    def _build_causal_mask(
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch, sequence = input_ids.shape
        blocked = torch.triu(
            torch.ones(sequence, sequence, dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        blocked = blocked[None, None, :, :].expand(batch, 1, sequence, sequence)

        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("attention_mask must match input_ids shape")
            # Padding blocks key columns; padded query outputs are ignored by callers.
            blocked = blocked | ~attention_mask[:, None, None, :].to(torch.bool)

        mask = torch.zeros((batch, 1, sequence, sequence), dtype=dtype, device=input_ids.device)
        return mask.masked_fill(blocked, torch.finfo(dtype).min)


@dataclass(frozen=True)
class Qwen3Output:
    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None = None


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: EngineConfig) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # One Parameter object guarantees true storage sharing, not equal copies.
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> Qwen3Output:
        hidden_states, captured = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=output_hidden_states,
        )
        logits = self.lm_head(hidden_states)
        return Qwen3Output(logits=logits, hidden_states=captured)

    def forward_cached(
        self,
        input_ids: torch.Tensor,
        *,
        cache: PagedKVCache,
        sequence_ids: Sequence[SequenceId],
        output_hidden_states: bool = False,
    ) -> Qwen3Output:
        """Run a transactional paged-cache append without branching dense forward."""

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if len(sequence_ids) != input_ids.shape[0]:
            raise ValueError("sequence_ids must contain one ID per input row")
        if cache.config != self.config:
            raise ValueError("cache and model must use the same EngineConfig")

        reservation = cache.begin_append(sequence_ids, input_ids.shape[1])
        position_ids = reservation.query_start_positions[:, None] + torch.arange(
            input_ids.shape[1], device=input_ids.device
        )[None, :]
        try:
            hidden_states, captured = self.model.forward_cached(
                input_ids,
                position_ids=position_ids,
                cache=cache,
                reservation=reservation,
                output_hidden_states=output_hidden_states,
            )
            logits = self.lm_head(hidden_states)
            cache.commit(reservation)
        except Exception:
            if reservation.state == "open":
                cache.rollback(reservation)
            raise
        return Qwen3Output(logits=logits, hidden_states=captured)


@dataclass(frozen=True)
class HiddenStateDifference:
    index: int
    max_absolute_error: float


def first_hidden_state_difference(
    reference: tuple[torch.Tensor, ...],
    candidate: tuple[torch.Tensor, ...],
    *,
    rtol: float,
    atol: float,
) -> HiddenStateDifference | None:
    """Return the first layer boundary that fails the declared tolerance."""

    if len(reference) != len(candidate):
        raise ValueError("hidden-state sequences have different lengths")
    for index, (expected, actual) in enumerate(zip(reference, candidate, strict=True)):
        if expected.shape != actual.shape:
            return HiddenStateDifference(index=index, max_absolute_error=math.inf)
        if not torch.allclose(expected, actual, rtol=rtol, atol=atol):
            max_error = (expected - actual).abs().max().item()
            return HiddenStateDifference(index=index, max_absolute_error=max_error)
    return None
