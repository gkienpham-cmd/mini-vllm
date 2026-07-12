from __future__ import annotations

import gc
from dataclasses import dataclass

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from engine.cache import PagedKVCache
from engine.config import CANONICAL_MODEL_ID, CANONICAL_MODEL_REVISION
from engine.generation import greedy_decode, paged_greedy_decode
from engine.model.loader import load_model, resolve_checkpoint
from engine.model.qwen3 import first_hidden_state_difference
from engine.scheduler import ContinuousBatchScheduler

MAX_NEW_TOKENS = 8


@dataclass(frozen=True)
class PromptFixture:
    category: str
    text: str


# Keep exactly five stable prompts; output text is never used as the parity oracle.
PROMPTS = (
    PromptFixture("short", "Hello"),
    PromptFixture(
        "long",
        "An inference engine receives several language-model requests with different "
        "prompt lengths. Explain why validating the mathematical forward pass must "
        "happen before optimizing memory allocation or request scheduling.",
    ),
    PromptFixture("punctuation-heavy", "Wait... what?! (Really); yes: 3, 2, 1—go!"),
    PromptFixture("repeated-token", "go go go go go go go go go go go go"),
    PromptFixture("boundary-sensitive", " token" * 31),
)
assert len(PROMPTS) == 5


@torch.inference_mode()
def _reference_greedy_decode(
    model,
    input_ids: torch.Tensor,
    *,
    eos_token_id: int | None,
) -> torch.Tensor:
    token_ids = input_ids
    for _ in range(MAX_NEW_TOKENS):
        next_token = model(token_ids).logits[:, -1].argmax(dim=-1)
        token_ids = torch.cat((token_ids, next_token[:, None]), dim=1)
        if eos_token_id is not None and bool(next_token.eq(eos_token_id).all()):
            break
    return token_ids


def _tokenize_prompts(tokenizer, device: torch.device) -> list[torch.Tensor]:
    return [
        tokenizer(
            fixture.text,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids.to(device)
        for fixture in PROMPTS
    ]


def _run_full_parity(device: torch.device, dtype: torch.dtype) -> None:
    checkpoint_dir = resolve_checkpoint(
        CANONICAL_MODEL_ID,
        revision=CANONICAL_MODEL_REVISION,
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    tokenized = _tokenize_prompts(tokenizer, device)

    reference = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        torch_dtype=dtype,
        attn_implementation="eager",
    ).to(device)
    reference.eval()
    expected_tokens = [
        _reference_greedy_decode(
            reference,
            input_ids,
            eos_token_id=tokenizer.eos_token_id,
        ).cpu()
        for input_ids in tokenized
    ]
    expected_hidden = tuple(
        state.detach().cpu().clone()
        for state in reference(tokenized[0], output_hidden_states=True).hidden_states
    )
    del reference
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    mini_dtype = "float16" if dtype == torch.float16 else "float32"
    model, report, _ = load_model(
        checkpoint_dir,
        revision=CANONICAL_MODEL_REVISION,
        device=str(device),
        dtype=mini_dtype,
        num_kv_blocks=8,
        max_num_batched_tokens=16,
    )
    assert report.consumed
    assert model.model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()

    actual_tokens = [
        greedy_decode(
            model,
            input_ids,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
        ).cpu()
        for input_ids in tokenized
    ]
    cache = PagedKVCache(model.config)
    cached_tokens = [
        paged_greedy_decode(
            model,
            input_ids,
            cache=cache,
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
            sequence_ids=[fixture.category],
        ).cpu()
        for fixture, input_ids in zip(PROMPTS, tokenized, strict=True)
    ]
    cache.assert_no_leaks()

    scheduler_cache = PagedKVCache(model.config)
    scheduled_tokens = []
    for fixture, input_ids in zip(PROMPTS, tokenized, strict=True):
        scheduler = ContinuousBatchScheduler(model, scheduler_cache)
        scheduler.submit(
            fixture.category,
            input_ids[0],
            max_new_tokens=MAX_NEW_TOKENS,
            eos_token_id=tokenizer.eos_token_id,
        )
        while scheduler.has_unfinished_requests:
            scheduler.step()
        generated = torch.tensor(
            [scheduler.get_request(fixture.category).generated_token_ids],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        scheduled_tokens.append(torch.cat((input_ids, generated), dim=1).cpu())
    scheduler_cache.assert_no_leaks()

    for fixture, expected, dense, cached, scheduled in zip(
        PROMPTS,
        expected_tokens,
        actual_tokens,
        cached_tokens,
        scheduled_tokens,
        strict=True,
    ):
        assert expected.shape == dense.shape == cached.shape == scheduled.shape, (
            fixture.category
        )
        # Comparing every column makes the first divergent decode step visible.
        for step in range(expected.shape[1]):
            torch.testing.assert_close(
                dense[:, step],
                expected[:, step],
                rtol=0.0,
                atol=0.0,
                msg=f"dense {fixture.category} diverged at token column {step}",
            )
            torch.testing.assert_close(
                cached[:, step],
                expected[:, step],
                rtol=0.0,
                atol=0.0,
                msg=f"paged {fixture.category} diverged at token column {step}",
            )
            torch.testing.assert_close(
                scheduled[:, step],
                expected[:, step],
                rtol=0.0,
                atol=0.0,
                msg=f"scheduler {fixture.category} diverged at token column {step}",
            )

    actual_output = model(tokenized[0], output_hidden_states=True)
    assert actual_output.hidden_states is not None
    actual_hidden = tuple(state.detach().cpu() for state in actual_output.hidden_states)
    rtol, atol = (1e-3, 1e-3) if dtype == torch.float16 else (1e-5, 1e-6)
    difference = first_hidden_state_difference(
        expected_hidden,
        actual_hidden,
        rtol=rtol,
        atol=atol,
    )
    assert difference is None, difference


@pytest.mark.model
def test_full_checkpoint_cpu_fp32_parity() -> None:
    _run_full_parity(torch.device("cpu"), torch.float32)


@pytest.mark.model
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_full_checkpoint_t4_fp16_parity() -> None:
    _run_full_parity(torch.device("cuda"), torch.float16)
