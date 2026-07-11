from __future__ import annotations

import torch
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention as HFQwen3Attention,
    Qwen3ForCausalLM as HFQwen3ForCausalLM,
    Qwen3MLP as HFQwen3MLP,
    Qwen3RMSNorm as HFQwen3RMSNorm,
    Qwen3RotaryEmbedding as HFQwen3RotaryEmbedding,
)

from engine.model.qwen3 import (
    Qwen3Attention,
    Qwen3ForCausalLM,
    Qwen3MLP,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    first_hidden_state_difference,
)

CPU_RTOL = 1e-5
CPU_ATOL = 1e-6


def _hf_config(tiny_config) -> Qwen3Config:
    config = Qwen3Config(
        vocab_size=tiny_config.vocab_size,
        hidden_size=tiny_config.hidden_size,
        intermediate_size=tiny_config.intermediate_size,
        num_hidden_layers=tiny_config.num_hidden_layers,
        num_attention_heads=tiny_config.num_attention_heads,
        num_key_value_heads=tiny_config.num_key_value_heads,
        head_dim=tiny_config.head_dim,
        max_position_embeddings=tiny_config.max_position_embeddings,
        rope_theta=tiny_config.rope_theta,
        rms_norm_eps=tiny_config.rms_norm_eps,
        hidden_act=tiny_config.hidden_act,
        attention_bias=tiny_config.attention_bias,
        attention_dropout=tiny_config.attention_dropout,
        tie_word_embeddings=tiny_config.tie_word_embeddings,
        bos_token_id=tiny_config.bos_token_id,
        eos_token_id=tiny_config.eos_token_id,
        use_cache=False,
    )
    config._attn_implementation = "eager"
    return config


def _causal_mask(batch: int, sequence: int) -> torch.Tensor:
    blocked = torch.triu(torch.ones(sequence, sequence, dtype=torch.bool), diagonal=1)
    mask = torch.zeros(batch, 1, sequence, sequence)
    return mask.masked_fill(blocked[None, None], torch.finfo(torch.float32).min)


def test_rmsnorm_matches_hugging_face(tiny_config) -> None:
    ours = Qwen3RMSNorm(tiny_config.hidden_size, tiny_config.rms_norm_eps)
    reference = HFQwen3RMSNorm(tiny_config.hidden_size, tiny_config.rms_norm_eps)
    ours.load_state_dict(reference.state_dict())
    inputs = torch.randn(2, 7, tiny_config.hidden_size)

    torch.testing.assert_close(
        ours(inputs), reference(inputs), rtol=CPU_RTOL, atol=CPU_ATOL
    )


def test_rope_matches_hugging_face(tiny_config) -> None:
    hf_config = _hf_config(tiny_config)
    ours = Qwen3RotaryEmbedding(tiny_config)
    reference = HFQwen3RotaryEmbedding(hf_config)
    inputs = torch.randn(2, 7, tiny_config.head_dim)
    position_ids = torch.arange(7)[None, :].expand(2, -1)

    ours_cos, ours_sin = ours(position_ids, dtype=inputs.dtype)
    reference_cos, reference_sin = reference(inputs, position_ids)

    torch.testing.assert_close(
        ours_cos.squeeze(1), reference_cos, rtol=CPU_RTOL, atol=CPU_ATOL
    )
    torch.testing.assert_close(
        ours_sin.squeeze(1), reference_sin, rtol=CPU_RTOL, atol=CPU_ATOL
    )


def test_swiglu_matches_hugging_face(tiny_config) -> None:
    hf_config = _hf_config(tiny_config)
    ours = Qwen3MLP(tiny_config)
    reference = HFQwen3MLP(hf_config)
    ours.load_state_dict(reference.state_dict())
    inputs = torch.randn(2, 7, tiny_config.hidden_size)

    torch.testing.assert_close(
        ours(inputs), reference(inputs), rtol=CPU_RTOL, atol=CPU_ATOL
    )


def test_gqa_qk_norm_attention_matches_hugging_face(tiny_config) -> None:
    hf_config = _hf_config(tiny_config)
    ours = Qwen3Attention(tiny_config, Qwen3RotaryEmbedding(tiny_config))
    reference = HFQwen3Attention(hf_config, layer_idx=0)
    ours.load_state_dict(reference.state_dict())
    inputs = torch.randn(2, 7, tiny_config.hidden_size)
    position_ids = torch.arange(7)[None, :].expand(2, -1)
    mask = _causal_mask(batch=2, sequence=7)
    hf_rope = HFQwen3RotaryEmbedding(hf_config)
    position_embeddings = hf_rope(inputs, position_ids)

    expected = reference(
        inputs,
        position_embeddings=position_embeddings,
        attention_mask=mask,
    )[0]
    actual = ours(inputs, position_ids=position_ids, attention_mask=mask)

    torch.testing.assert_close(
        actual, expected, rtol=CPU_RTOL, atol=CPU_ATOL
    )


def test_tiny_full_model_and_layer_boundaries_match_hugging_face(tiny_config) -> None:
    hf_config = _hf_config(tiny_config)
    reference = HFQwen3ForCausalLM(hf_config).eval()
    ours = Qwen3ForCausalLM(tiny_config).eval()
    ours.load_state_dict(reference.state_dict(), strict=True)
    input_ids = torch.tensor([[1, 8, 3, 9, 2], [1, 4, 4, 5, 2]])

    expected = reference(input_ids, output_hidden_states=True)
    actual = ours(input_ids, output_hidden_states=True)

    torch.testing.assert_close(
        actual.logits, expected.logits, rtol=CPU_RTOL, atol=CPU_ATOL
    )
    assert actual.hidden_states is not None
    assert expected.hidden_states is not None
    difference = first_hidden_state_difference(
        expected.hidden_states,
        actual.hidden_states,
        rtol=CPU_RTOL,
        atol=CPU_ATOL,
    )
    assert difference is None


def test_layer_diagnostic_returns_first_failure() -> None:
    reference = (torch.zeros(2), torch.zeros(2), torch.zeros(2))
    candidate = (torch.zeros(2), torch.ones(2), torch.full((2,), 2.0))

    difference = first_hidden_state_difference(
        reference,
        candidate,
        rtol=CPU_RTOL,
        atol=CPU_ATOL,
    )

    assert difference is not None
    assert difference.index == 1
    assert difference.max_absolute_error == 1.0

