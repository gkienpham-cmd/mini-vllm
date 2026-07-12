"""Produce deterministic Milestone 2 cache correctness evidence as JSON."""

from __future__ import annotations

import json
import math
from typing import Any

import torch

from engine.cache import PagedKVCache
from engine.config import EngineConfig
from engine.model.qwen3 import Qwen3ForCausalLM

PROMPT_LENGTHS = (1, 15, 16, 17, 31)
CPU_RTOL = 1e-5
# Cached one-token and dense full-prefix GEMMs reduce FP32 values in different orders.
INTERNAL_ATOL = 2e-6


def _tiny_config() -> EngineConfig:
    return EngineConfig(
        model_id="tiny-qwen3-cache-evidence",
        revision="milestone-2",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rope_theta=1_000_000.0,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        attention_bias=False,
        attention_dropout=0.0,
        tie_word_embeddings=True,
        bos_token_id=1,
        eos_token_id=2,
        num_kv_blocks=8,
    )


def _cache_bytes(config: EngineConfig) -> int:
    element_size = torch.empty((), dtype=config.torch_dtype).element_size()
    return (
        config.num_hidden_layers
        * 2
        * config.num_kv_blocks
        * config.kv_block_size
        * config.num_key_value_heads
        * config.head_dim
        * element_size
    )


def collect_evidence() -> dict[str, Any]:
    torch.manual_seed(123)
    torch.use_deterministic_algorithms(True)
    config = _tiny_config()
    model = Qwen3ForCausalLM(config).eval()
    cases: list[dict[str, Any]] = []

    with torch.inference_mode():
        for prompt_length in PROMPT_LENGTHS:
            cache = PagedKVCache(config)
            cache.create_sequence(prompt_length)
            prompt = torch.arange(prompt_length).remainder(config.vocab_size)[None, :]
            model.forward_cached(
                prompt, cache=cache, sequence_ids=[prompt_length]
            )
            next_input = torch.tensor([[7]])
            dense = model(torch.cat((prompt, next_input), dim=1)).logits[:, -1]
            cached = model.forward_cached(
                next_input, cache=cache, sequence_ids=[prompt_length]
            ).logits[:, -1]
            difference = (dense - cached).abs()
            relative = difference / dense.abs().clamp_min(torch.finfo(dense.dtype).tiny)
            cases.append(
                {
                    "prompt_length": prompt_length,
                    "physical_blocks": math.ceil(prompt_length / config.kv_block_size),
                    "unused_tail_slots": (
                        -prompt_length % config.kv_block_size
                    ),
                    "max_absolute_logit_error": difference.max().item(),
                    "max_relative_logit_error": relative.max().item(),
                    "within_internal_tolerance": torch.allclose(
                        dense, cached, rtol=CPU_RTOL, atol=INTERNAL_ATOL
                    ),
                    "greedy_token_equal": bool(
                        dense.argmax(dim=-1).eq(cached.argmax(dim=-1)).all()
                    ),
                }
            )
            cache.release_sequence(prompt_length)
            cache.assert_no_leaks()

    canonical_fp32_bytes_for_8_blocks = (
        EngineConfig.CANONICAL_QWEN3_06B["num_hidden_layers"]
        * 2
        * config.num_kv_blocks
        * config.kv_block_size
        * EngineConfig.CANONICAL_QWEN3_06B["num_key_value_heads"]
        * EngineConfig.CANONICAL_QWEN3_06B["head_dim"]
        * torch.empty((), dtype=torch.float32).element_size()
    )
    return {
        "seed": 123,
        "dtype": config.dtype,
        "kv_block_size": config.kv_block_size,
        "num_kv_blocks": config.num_kv_blocks,
        "tiny_cache_bytes": _cache_bytes(config),
        "canonical_fp32_cache_bytes_for_8_blocks": canonical_fp32_bytes_for_8_blocks,
        "rtol": CPU_RTOL,
        "atol": INTERNAL_ATOL,
        "cases": cases,
    }


if __name__ == "__main__":
    print(json.dumps(collect_evidence(), indent=2, sort_keys=True))
