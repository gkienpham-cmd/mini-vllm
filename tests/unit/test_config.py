from __future__ import annotations

import pytest

from engine.config import CANONICAL_MODEL_REVISION, EngineConfig


def test_checkpoint_values_flow_into_one_config() -> None:
    raw = dict(EngineConfig.CANONICAL_QWEN3_06B)
    raw.update(bos_token_id=151643, eos_token_id=151643)

    config = EngineConfig.from_hf_config(raw)

    assert config.model_id == "Qwen/Qwen3-0.6B-Base"
    assert config.revision == CANONICAL_MODEL_REVISION
    assert config.query_projection_size == 2048
    assert config.key_value_projection_size == 1024
    assert config.query_heads_per_kv_head == 2
    assert config.kv_block_size == 16
    assert config.num_kv_blocks == 0
    assert config.max_num_batched_tokens == 0


def test_canonical_validation_rejects_silent_architecture_drift() -> None:
    raw = dict(EngineConfig.CANONICAL_QWEN3_06B)
    raw.update(bos_token_id=151643, eos_token_id=151643, num_attention_heads=8)

    with pytest.raises(ValueError, match="num_attention_heads"):
        EngineConfig.from_hf_config(raw)


def test_cpu_rejects_non_fp32_runtime(tiny_config: EngineConfig) -> None:
    values = dict(tiny_config.__dict__)
    values["dtype"] = "float16"

    with pytest.raises(ValueError, match="CPU is the FP32"):
        EngineConfig(**values)


def test_cache_runtime_choices_are_validated(tiny_config: EngineConfig) -> None:
    values = dict(tiny_config.__dict__)
    values["kv_block_size"] = 32
    with pytest.raises(ValueError, match="block size of 16"):
        EngineConfig(**values)

    values = dict(tiny_config.__dict__)
    values["num_kv_blocks"] = -1
    with pytest.raises(ValueError, match="non-negative"):
        EngineConfig(**values)

    values = dict(tiny_config.__dict__)
    values["max_num_batched_tokens"] = -1
    with pytest.raises(ValueError, match="max_num_batched_tokens"):
        EngineConfig(**values)
