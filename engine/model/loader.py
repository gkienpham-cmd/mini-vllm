"""Strict safetensors loading with complete key and shape accounting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

from engine.config import CANONICAL_MODEL_ID, CANONICAL_MODEL_REVISION, EngineConfig
from engine.model.qwen3 import Qwen3ForCausalLM


@dataclass(frozen=True)
class WeightLoadReport:
    consumed: tuple[str, ...]
    tied_aliases: tuple[str, ...]
    checkpoint_files: tuple[str, ...]


def resolve_checkpoint(
    checkpoint: str | Path,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    local_path = Path(checkpoint)
    if local_path.exists():
        return local_path

    # Downloading is transport only; Transformers never participates in model math.
    downloaded = snapshot_download(
        repo_id=str(checkpoint),
        revision=revision,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        allow_patterns=[
            "config.json",
            "model*.safetensors",
            "model.safetensors.index.json",
            "tokenizer*",
            "vocab*",
            "merges.txt",
            "special_tokens_map.json",
        ],
    )
    return Path(downloaded)


def _checkpoint_files(checkpoint_dir: Path) -> tuple[Path, ...]:
    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        with index_path.open(encoding="utf-8") as index_file:
            weight_map = json.load(index_file)["weight_map"]
        return tuple(sorted({checkpoint_dir / name for name in weight_map.values()}))

    single_file = checkpoint_dir / "model.safetensors"
    if single_file.exists():
        return (single_file,)
    raise FileNotFoundError(f"no safetensors checkpoint found in {checkpoint_dir}")


def _tensor_entries(files: tuple[Path, ...]) -> Iterator[tuple[str, Path]]:
    for checkpoint_file in files:
        with safe_open(checkpoint_file, framework="pt", device="cpu") as tensors:
            for key in tensors.keys():
                yield key, checkpoint_file


@torch.no_grad()
def load_safetensors(
    model: Qwen3ForCausalLM,
    checkpoint_dir: str | Path,
) -> WeightLoadReport:
    checkpoint_path = Path(checkpoint_dir)
    files = _checkpoint_files(checkpoint_path)
    destinations = model.state_dict()
    entries = list(_tensor_entries(files))
    checkpoint_keys = {key for key, _ in entries}
    expected_keys = set(destinations)

    # A tied checkpoint may store only the embedding copy of lm_head.weight.
    optional_tied_key = "lm_head.weight"
    required_keys = expected_keys - {optional_tied_key}
    missing = sorted(required_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(
            f"checkpoint key mismatch: missing={missing}, unexpected={unexpected}"
        )

    consumed: list[str] = []
    entry_files = dict(entries)
    for checkpoint_file in files:
        with safe_open(checkpoint_file, framework="pt", device="cpu") as tensors:
            for key in sorted(set(tensors.keys()) - {optional_tied_key}):
                source = tensors.get_tensor(key)
                destination = destinations[key]
                if source.shape != destination.shape:
                    raise ValueError(
                        f"shape mismatch for {key}: checkpoint={tuple(source.shape)}, "
                        f"model={tuple(destination.shape)}"
                    )
                # Both formats use [out_features, in_features]; never transpose.
                destination.copy_(source.to(dtype=destination.dtype))
                consumed.append(key)

    tied_aliases: list[str] = []
    if optional_tied_key in checkpoint_keys:
        with safe_open(
            entry_files[optional_tied_key], framework="pt", device="cpu"
        ) as tensors:
            tied_source = tensors.get_tensor(optional_tied_key)
        tied_destination = destinations[optional_tied_key]
        if tied_source.shape != tied_destination.shape:
            raise ValueError(
                f"shape mismatch for {optional_tied_key}: "
                f"checkpoint={tuple(tied_source.shape)}, "
                f"model={tuple(tied_destination.shape)}"
            )
        if not torch.equal(tied_source.to(tied_destination.dtype), tied_destination):
            raise ValueError("lm_head.weight disagrees with tied embedding weights")
        consumed.append(optional_tied_key)
    else:
        tied_aliases.append(optional_tied_key)

    return WeightLoadReport(
        consumed=tuple(sorted(consumed)),
        tied_aliases=tuple(tied_aliases),
        checkpoint_files=tuple(str(path) for path in files),
    )


def load_model(
    checkpoint: str | Path = CANONICAL_MODEL_ID,
    *,
    revision: str | None = None,
    cache_dir: str | Path | None = None,
    device: str = "cpu",
    dtype: str = "float32",
    kv_block_size: int = 16,
    num_kv_blocks: int = 0,
    require_canonical: bool = True,
) -> tuple[Qwen3ForCausalLM, WeightLoadReport, Path]:
    resolved_revision = revision
    if str(checkpoint) == CANONICAL_MODEL_ID and resolved_revision is None:
        # Pinning prevents a future Hub update from silently changing parity.
        resolved_revision = CANONICAL_MODEL_REVISION
    checkpoint_dir = resolve_checkpoint(
        checkpoint,
        revision=resolved_revision,
        cache_dir=cache_dir,
    )
    config = EngineConfig.from_json(
        checkpoint_dir / "config.json",
        model_id=str(checkpoint),
        revision=resolved_revision,
        device=device,
        dtype=dtype,
        kv_block_size=kv_block_size,
        num_kv_blocks=num_kv_blocks,
        require_canonical=require_canonical,
    )

    # Approved Milestone 1 choice: construct plainly on CPU for debuggability.
    model = Qwen3ForCausalLM(config)
    report = load_safetensors(model, checkpoint_dir)
    model.to(device=torch.device(device), dtype=config.torch_dtype)
    model.eval()
    return model, report, checkpoint_dir
