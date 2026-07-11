from __future__ import annotations

import os
import random

import pytest
import torch

from engine.config import EngineConfig

# Deterministic CUDA matmuls require this before a CUDA context is initialized.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@pytest.fixture(autouse=True)
def deterministic_test_state() -> None:
    random.seed(123)
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)
    torch.use_deterministic_algorithms(True)


@pytest.fixture
def tiny_config() -> EngineConfig:
    return EngineConfig(
        model_id="tiny-qwen3",
        revision="test",
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
        device="cpu",
        dtype="float32",
    )
