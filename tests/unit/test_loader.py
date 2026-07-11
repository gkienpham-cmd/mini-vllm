from __future__ import annotations

from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from engine.model.loader import load_safetensors
from engine.model.qwen3 import Qwen3ForCausalLM


def _write_checkpoint(model: Qwen3ForCausalLM, directory: Path) -> None:
    # Safetensors stores one copy of tied parameters, just like the HF checkpoint.
    weights = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if key != "lm_head.weight"
    }
    save_file(weights, directory / "model.safetensors")


def test_loader_consumes_every_weight_and_preserves_tying(tmp_path, tiny_config) -> None:
    source = Qwen3ForCausalLM(tiny_config)
    with torch.no_grad():
        for index, parameter in enumerate(source.parameters()):
            parameter.fill_(index / 100.0)
    _write_checkpoint(source, tmp_path)

    target = Qwen3ForCausalLM(tiny_config)
    report = load_safetensors(target, tmp_path)

    assert report.tied_aliases == ("lm_head.weight",)
    assert target.model.embed_tokens.weight.data_ptr() == target.lm_head.weight.data_ptr()
    for key, expected in source.state_dict().items():
        torch.testing.assert_close(
            target.state_dict()[key], expected, rtol=0.0, atol=0.0
        )


def test_loader_rejects_wrong_shape_without_transposing(tmp_path, tiny_config) -> None:
    source = Qwen3ForCausalLM(tiny_config)
    weights = {
        key: value.detach().clone()
        for key, value in source.state_dict().items()
        if key != "lm_head.weight"
    }
    weights["model.layers.0.mlp.down_proj.weight"] = weights[
        "model.layers.0.mlp.down_proj.weight"
    ].transpose(0, 1).contiguous()
    save_file(weights, tmp_path / "model.safetensors")

    with pytest.raises(ValueError, match="shape mismatch"):
        load_safetensors(Qwen3ForCausalLM(tiny_config), tmp_path)


def test_loader_rejects_unexpected_tensor(tmp_path, tiny_config) -> None:
    source = Qwen3ForCausalLM(tiny_config)
    weights = {
        key: value.detach().clone()
        for key, value in source.state_dict().items()
        if key != "lm_head.weight"
    }
    weights["unexpected.weight"] = torch.zeros(1)
    save_file(weights, tmp_path / "model.safetensors")

    with pytest.raises(ValueError, match="unexpected.weight"):
        load_safetensors(Qwen3ForCausalLM(tiny_config), tmp_path)
