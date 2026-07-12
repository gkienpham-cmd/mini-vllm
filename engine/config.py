"""The single shared configuration object used by every engine subsystem."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, Mapping

import torch

CANONICAL_MODEL_ID = "Qwen/Qwen3-0.6B-Base"
CANONICAL_MODEL_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"


@dataclass(frozen=True)
class EngineConfig:
    """Architecture and runtime settings for one mini-vllm engine instance.

    Checkpoint-derived fields live beside runtime fields so downstream modules
    cannot quietly disagree about tensor shapes, device, or dtype.
    """

    model_id: str
    revision: str | None
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rope_theta: float
    rms_norm_eps: float
    hidden_act: str
    attention_bias: bool
    attention_dropout: float
    tie_word_embeddings: bool
    bos_token_id: int | None
    eos_token_id: int | None
    device: str = "cpu"
    dtype: str = "float32"
    kv_block_size: int = 16
    num_kv_blocks: int = 0

    # This signature is centralized here; model modules never duplicate it.
    CANONICAL_QWEN3_06B: ClassVar[Mapping[str, Any]] = {
        "model_type": "qwen3",
        "vocab_size": 151936,
        "hidden_size": 1024,
        "intermediate_size": 3072,
        "num_hidden_layers": 28,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 32768,
        "rope_theta": 1_000_000.0,
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": True,
    }

    def __post_init__(self) -> None:
        if self.hidden_size <= 0 or self.head_dim <= 0:
            raise ValueError("hidden_size and head_dim must be positive")
        if self.num_attention_heads <= 0 or self.num_key_value_heads <= 0:
            raise ValueError("attention head counts must be positive")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("query heads must divide evenly across KV heads")
        if self.head_dim % 2 != 0:
            raise ValueError("RoPE requires an even head_dim")
        if self.hidden_act != "silu":
            raise ValueError("Milestone 1 supports Qwen3's SiLU activation only")
        if self.attention_bias:
            raise ValueError("Milestone 1 supports bias-free Qwen3 attention only")
        if self.attention_dropout != 0.0:
            raise ValueError("inference requires zero attention dropout")
        if not self.tie_word_embeddings:
            raise ValueError("Milestone 1 requires Qwen3-0.6B's tied embeddings")
        if self.dtype not in {"float32", "float16"}:
            raise ValueError("dtype must be 'float32' or 'float16'")
        if self.device == "cpu" and self.dtype != "float32":
            raise ValueError("CPU is the FP32 correctness path")
        if self.kv_block_size != 16:
            raise ValueError("Milestone 2 requires a KV block size of 16 tokens")
        if self.num_kv_blocks < 0:
            raise ValueError("num_kv_blocks must be non-negative")

    @property
    def torch_dtype(self) -> torch.dtype:
        return {"float32": torch.float32, "float16": torch.float16}[self.dtype]

    @property
    def query_projection_size(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def key_value_projection_size(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def query_heads_per_kv_head(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_json(
        cls,
        config_path: str | Path,
        *,
        model_id: str = CANONICAL_MODEL_ID,
        revision: str | None = CANONICAL_MODEL_REVISION,
        device: str = "cpu",
        dtype: str = "float32",
        kv_block_size: int = 16,
        num_kv_blocks: int = 0,
        require_canonical: bool = True,
    ) -> "EngineConfig":
        with Path(config_path).open(encoding="utf-8") as config_file:
            checkpoint_config = json.load(config_file)
        return cls.from_hf_config(
            checkpoint_config,
            model_id=model_id,
            revision=revision,
            device=device,
            dtype=dtype,
            kv_block_size=kv_block_size,
            num_kv_blocks=num_kv_blocks,
            require_canonical=require_canonical,
        )

    @classmethod
    def from_hf_config(
        cls,
        checkpoint_config: Mapping[str, Any],
        *,
        model_id: str = CANONICAL_MODEL_ID,
        revision: str | None = CANONICAL_MODEL_REVISION,
        device: str = "cpu",
        dtype: str = "float32",
        kv_block_size: int = 16,
        num_kv_blocks: int = 0,
        require_canonical: bool = True,
    ) -> "EngineConfig":
        if require_canonical:
            mismatches = {
                key: (checkpoint_config.get(key), expected)
                for key, expected in cls.CANONICAL_QWEN3_06B.items()
                if checkpoint_config.get(key) != expected
            }
            if mismatches:
                details = ", ".join(
                    f"{key}={actual!r}, expected {expected!r}"
                    for key, (actual, expected) in sorted(mismatches.items())
                )
                raise ValueError(f"unsupported checkpoint architecture: {details}")

        required = (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "max_position_embeddings",
            "rope_theta",
            "rms_norm_eps",
            "hidden_act",
            "attention_bias",
            "attention_dropout",
            "tie_word_embeddings",
        )
        missing = [key for key in required if key not in checkpoint_config]
        if missing:
            raise ValueError(f"checkpoint config is missing: {', '.join(missing)}")

        return cls(
            model_id=model_id,
            revision=revision,
            vocab_size=int(checkpoint_config["vocab_size"]),
            hidden_size=int(checkpoint_config["hidden_size"]),
            intermediate_size=int(checkpoint_config["intermediate_size"]),
            num_hidden_layers=int(checkpoint_config["num_hidden_layers"]),
            num_attention_heads=int(checkpoint_config["num_attention_heads"]),
            num_key_value_heads=int(checkpoint_config["num_key_value_heads"]),
            head_dim=int(checkpoint_config["head_dim"]),
            max_position_embeddings=int(checkpoint_config["max_position_embeddings"]),
            rope_theta=float(checkpoint_config["rope_theta"]),
            rms_norm_eps=float(checkpoint_config["rms_norm_eps"]),
            hidden_act=str(checkpoint_config["hidden_act"]),
            attention_bias=bool(checkpoint_config["attention_bias"]),
            attention_dropout=float(checkpoint_config["attention_dropout"]),
            tie_word_embeddings=bool(checkpoint_config["tie_word_embeddings"]),
            bos_token_id=checkpoint_config.get("bos_token_id"),
            eos_token_id=checkpoint_config.get("eos_token_id"),
            device=device,
            dtype=dtype,
            kv_block_size=kv_block_size,
            num_kv_blocks=num_kv_blocks,
        )
