from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from engine.cache import PagedKVCache
from engine.generation import greedy_decode
from engine.model.qwen3 import Qwen3ForCausalLM
from engine.scheduler import (
    ContinuousBatchScheduler,
    FinishReason,
    RequestStatus,
    SchedulerExecutionError,
)


def _scheduler(tiny_config, *, blocks: int = 8, budget: int = 4):
    config = replace(
        tiny_config,
        num_kv_blocks=blocks,
        max_num_batched_tokens=budget,
    )
    model = Qwen3ForCausalLM(config).eval()
    cache = PagedKVCache(config)
    return model, cache, ContinuousBatchScheduler(model, cache)


def _run_to_completion(scheduler: ContinuousBatchScheduler, limit: int = 256):
    steps = []
    for _ in range(limit):
        if not scheduler.has_unfinished_requests:
            return steps
        steps.append(scheduler.step())
    raise AssertionError("scheduler did not finish within the step limit")


def test_scheduler_requires_runtime_capacity(tiny_config) -> None:
    config = replace(tiny_config, num_kv_blocks=1)
    model = Qwen3ForCausalLM(config).eval()
    cache = PagedKVCache(config)
    with pytest.raises(ValueError, match="positive token budget"):
        ContinuousBatchScheduler(model, cache)


def test_submit_validation_and_zero_token_completion(tiny_config) -> None:
    _, cache, scheduler = _scheduler(tiny_config, blocks=2, budget=2)

    zero = scheduler.submit("zero", [1], max_new_tokens=0)
    assert zero.status is RequestStatus.FINISHED
    assert zero.finish_reason is FinishReason.LENGTH
    assert not cache.has_sequence("zero")

    invalid_cases = (
        ("empty", [], 1, "cannot be empty"),
        ("negative", [1], -1, "non-negative"),
        ("token", [tiny_config.vocab_size], 1, "vocabulary"),
        ("context", [1] * 64, 1, "context length"),
        ("cache", [1] * 32, 1, "physical KV cache"),
    )
    for request_id, prompt, max_new_tokens, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            scheduler.submit(
                request_id,
                prompt,
                max_new_tokens=max_new_tokens,
            )

    scheduler.submit("duplicate", [1], max_new_tokens=1)
    with pytest.raises(ValueError, match="already exists"):
        scheduler.submit("duplicate", [1], max_new_tokens=1)


def test_chunked_prefill_obeys_budget_and_matches_dense_greedy(tiny_config) -> None:
    model, cache, scheduler = _scheduler(tiny_config, blocks=8, budget=2)
    prompts = {
        "long": torch.tensor([[1, 5, 9, 3, 7]]),
        "short": torch.tensor([[4]]),
    }
    scheduler.submit("long", prompts["long"][0], max_new_tokens=3)
    scheduler.submit("short", prompts["short"][0], max_new_tokens=2)

    steps = _run_to_completion(scheduler)

    assert all(0 < step.scheduled_token_count <= 2 for step in steps)
    assert any("short" in step.admitted_request_ids for step in steps)
    for request_id, prompt in prompts.items():
        expected = greedy_decode(
            model,
            prompt,
            max_new_tokens=scheduler.get_request(request_id).max_new_tokens,
        )
        assert scheduler.get_request(request_id).generated_token_ids == expected[
            0, prompt.shape[1] :
        ].tolist()
    cache.assert_no_leaks()


def test_short_request_joins_and_retires_while_long_request_runs(tiny_config) -> None:
    _, cache, scheduler = _scheduler(tiny_config, blocks=6, budget=4)
    scheduler.submit("long", [1, 5, 9], max_new_tokens=6)
    first_step = scheduler.step()
    assert first_step.outputs[0].request_id == "long"
    assert scheduler.get_request("long").status is RequestStatus.RUNNING

    scheduler.submit("short", [4], max_new_tokens=1)
    second_step = scheduler.step()

    assert [output.request_id for output in second_step.outputs] == ["long", "short"]
    assert scheduler.get_request("short").status is RequestStatus.FINISHED
    assert scheduler.get_request("long").status is RequestStatus.RUNNING
    _run_to_completion(scheduler)
    cache.assert_no_leaks()


def test_repeated_arrivals_do_not_starve_oldest_running_request(tiny_config) -> None:
    _, cache, scheduler = _scheduler(tiny_config, blocks=4, budget=2)
    scheduler.submit("oldest", [1], max_new_tokens=6)
    scheduler.step()

    generated_counts = []
    for index in range(5):
        scheduler.submit(f"late-{index}", [index + 2], max_new_tokens=1)
        scheduler.step()
        generated_counts.append(
            len(scheduler.get_request("oldest").generated_token_ids)
        )

    assert generated_counts == [2, 3, 4, 5, 6]
    assert scheduler.get_request("oldest").status is RequestStatus.FINISHED
    _run_to_completion(scheduler)
    cache.assert_no_leaks()


def test_recompute_preemption_preserves_fifo_and_does_not_duplicate_tokens(
    tiny_config,
) -> None:
    model, cache, scheduler = _scheduler(tiny_config, blocks=3, budget=2)
    prompt = torch.arange(16).remainder(tiny_config.vocab_size)[None, :]
    scheduler.submit("old", prompt[0], max_new_tokens=20)
    scheduler.submit("new", prompt[0], max_new_tokens=4)

    steps = _run_to_completion(scheduler)
    preemptions = [
        request_id
        for step in steps
        for request_id in step.preempted_request_ids
    ]

    assert preemptions == ["new"]
    for request_id in ("old", "new"):
        request = scheduler.get_request(request_id)
        expected = greedy_decode(
            model,
            prompt,
            max_new_tokens=request.max_new_tokens,
        )
        assert request.generated_token_ids == expected[0, 16:].tolist()
        assert request.status is RequestStatus.FINISHED
    cache.assert_no_leaks()


def test_eos_retires_request_immediately(tiny_config) -> None:
    model, cache, scheduler = _scheduler(tiny_config, blocks=2, budget=2)
    prompt = torch.tensor([[1, 5]])
    eos_token_id = int(model(prompt).logits[0, -1].argmax())
    scheduler.submit(
        "request",
        prompt[0],
        max_new_tokens=4,
        eos_token_id=eos_token_id,
    )

    step = scheduler.step()

    assert step.outputs[0].finish_reason is FinishReason.EOS
    assert scheduler.get_request("request").generated_token_ids == [eos_token_id]
    assert scheduler.get_request("request").status is RequestStatus.FINISHED
    cache.assert_no_leaks()


def test_prefill_failure_marks_request_failed_and_releases_blocks(tiny_config) -> None:
    model, cache, scheduler = _scheduler(tiny_config, blocks=2, budget=2)
    scheduler.submit("request", [1, 5], max_new_tokens=2)

    with patch.object(model, "forward_cached", side_effect=RuntimeError("prefill failed")):
        with pytest.raises(SchedulerExecutionError, match="prefill failed") as caught:
            scheduler.step()

    request = scheduler.get_request("request")
    assert request.status is RequestStatus.FAILED
    assert request.finish_reason is FinishReason.ERROR
    assert request.error == "prefill failed"
    assert caught.value.step.outputs[0].finish_reason is FinishReason.ERROR
    assert caught.value.step.scheduled_token_count == 0
    cache.assert_no_leaks()


def test_batched_decode_failure_cleans_only_affected_requests(tiny_config) -> None:
    model, cache, scheduler = _scheduler(tiny_config, blocks=4, budget=4)
    scheduler.submit("first", [1], max_new_tokens=3)
    scheduler.submit("second", [2], max_new_tokens=3)
    scheduler.submit("unrelated", [3, 4, 5, 6, 7], max_new_tokens=1)
    scheduler.step()
    assert scheduler.get_request("first").status is RequestStatus.RUNNING
    assert scheduler.get_request("second").status is RequestStatus.RUNNING
    assert scheduler.get_request("unrelated").status is RequestStatus.PREFILLING

    with patch.object(model, "forward_cached", side_effect=RuntimeError("decode failed")):
        with pytest.raises(SchedulerExecutionError, match="decode failed") as caught:
            scheduler.step()

    assert scheduler.get_request("first").status is RequestStatus.FAILED
    assert scheduler.get_request("second").status is RequestStatus.FAILED
    assert scheduler.get_request("unrelated").status is RequestStatus.PREFILLING
    assert cache.has_sequence("unrelated")
    assert [output.request_id for output in caught.value.step.outputs] == [
        "first",
        "second",
    ]
    assert all(
        output.finish_reason is FinishReason.ERROR
        for output in caught.value.step.outputs
    )
    _run_to_completion(scheduler)
    cache.assert_no_leaks()
