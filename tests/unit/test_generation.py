from __future__ import annotations

import torch

from engine.generation import greedy_decode
from engine.model.qwen3 import Qwen3ForCausalLM


def test_greedy_decode_matches_manual_argmax(tiny_config) -> None:
    model = Qwen3ForCausalLM(tiny_config).eval()
    input_ids = torch.tensor([[1, 5, 9]], dtype=torch.long)

    expected_next = model(input_ids).logits[:, -1].argmax(dim=-1)
    actual = greedy_decode(model, input_ids, max_new_tokens=1)

    torch.testing.assert_close(actual[:, :-1], input_ids, rtol=0, atol=0)
    torch.testing.assert_close(actual[:, -1], expected_next, rtol=0, atol=0)


def test_greedy_decode_stops_after_eos(tiny_config) -> None:
    model = Qwen3ForCausalLM(tiny_config).eval()
    input_ids = torch.tensor([[1, 5]], dtype=torch.long)
    eos_token_id = int(model(input_ids).logits[:, -1].argmax())

    actual = greedy_decode(
        model,
        input_ids,
        max_new_tokens=4,
        eos_token_id=eos_token_id,
    )

    assert actual.shape[1] == input_ids.shape[1] + 1
    assert int(actual[0, -1]) == eos_token_id

