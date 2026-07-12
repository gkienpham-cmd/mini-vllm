"""Produce deterministic Milestone 3 scheduler correctness evidence as JSON."""

from __future__ import annotations

import json
from typing import Any

import torch

from engine.cache import PagedKVCache
from engine.config import EngineConfig
from engine.generation import greedy_decode
from engine.model.qwen3 import Qwen3ForCausalLM
from engine.scheduler import ContinuousBatchScheduler, SchedulerStep


def _tiny_config(*, blocks: int, budget: int) -> EngineConfig:
    return EngineConfig(
        model_id="tiny-qwen3-scheduler-evidence",
        revision="milestone-3",
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
        num_kv_blocks=blocks,
        max_num_batched_tokens=budget,
    )


def _serialize_step(
    index: int,
    step: SchedulerStep,
    scheduler: ContinuousBatchScheduler,
) -> dict[str, Any]:
    return {
        "step": index,
        "scheduled_token_count": step.scheduled_token_count,
        "admitted_request_ids": list(step.admitted_request_ids),
        "preempted_request_ids": list(step.preempted_request_ids),
        "outputs": [
            {
                "request_id": output.request_id,
                "token_id": output.token_id,
                "status": output.status.value,
                "finish_reason": (
                    output.finish_reason.value
                    if output.finish_reason is not None
                    else None
                ),
            }
            for output in step.outputs
        ],
        "request_states": {
            str(request_id): scheduler.get_request(request_id).status.value
            for request_id in scheduler.request_ids
        },
        "free_block_count": scheduler.cache.free_block_count,
    }


def _generated_matches_dense(
    model: Qwen3ForCausalLM,
    scheduler: ContinuousBatchScheduler,
    prompts: dict[str, torch.Tensor],
) -> dict[str, bool]:
    matches = {}
    for request_id, prompt in prompts.items():
        request = scheduler.get_request(request_id)
        dense = greedy_decode(
            model,
            prompt,
            max_new_tokens=request.max_new_tokens,
        )
        matches[request_id] = (
            request.generated_token_ids == dense[0, prompt.shape[1] :].tolist()
        )
    return matches


def _continuous_admission_evidence() -> dict[str, Any]:
    torch.manual_seed(123)
    config = _tiny_config(blocks=6, budget=4)
    model = Qwen3ForCausalLM(config).eval()
    cache = PagedKVCache(config)
    scheduler = ContinuousBatchScheduler(model, cache)
    prompts = {
        "long": torch.tensor([[1, 5, 9]]),
        "short": torch.tensor([[4]]),
    }
    scheduler.submit("long", prompts["long"][0], max_new_tokens=6)

    trace = []
    first_step = scheduler.step()
    trace.append(_serialize_step(1, first_step, scheduler))
    scheduler.submit("short", prompts["short"][0], max_new_tokens=1)
    step_index = 2
    while scheduler.has_unfinished_requests:
        step = scheduler.step()
        trace.append(_serialize_step(step_index, step, scheduler))
        step_index += 1

    completion_steps = {
        output["request_id"]: step["step"]
        for step in trace
        for output in step["outputs"]
        if output["finish_reason"] is not None
    }
    dense_matches = _generated_matches_dense(model, scheduler, prompts)
    cache.assert_no_leaks()
    return {
        "name": "continuous_admission",
        "token_budget": config.max_num_batched_tokens,
        "trace": trace,
        "max_scheduled_token_count": max(
            step["scheduled_token_count"] for step in trace
        ),
        "budget_respected": all(
            step["scheduled_token_count"] <= config.max_num_batched_tokens
            for step in trace
        ),
        "completion_steps": completion_steps,
        "short_finished_while_long_active": (
            completion_steps["short"] < completion_steps["long"]
        ),
        "dense_token_match": dense_matches,
        "leak_free": True,
    }


def _preemption_evidence() -> dict[str, Any]:
    torch.manual_seed(123)
    config = _tiny_config(blocks=3, budget=2)
    model = Qwen3ForCausalLM(config).eval()
    cache = PagedKVCache(config)
    scheduler = ContinuousBatchScheduler(model, cache)
    prompt = torch.arange(16).remainder(config.vocab_size)[None, :]
    prompts = {"old": prompt, "new": prompt}
    scheduler.submit("old", prompt[0], max_new_tokens=20)
    scheduler.submit("new", prompt[0], max_new_tokens=4)

    trace = []
    step_index = 1
    while scheduler.has_unfinished_requests:
        step = scheduler.step()
        trace.append(_serialize_step(step_index, step, scheduler))
        step_index += 1

    preemption_order = [
        request_id
        for step in trace
        for request_id in step["preempted_request_ids"]
    ]
    completion_order = [
        output["request_id"]
        for step in trace
        for output in step["outputs"]
        if output["finish_reason"] is not None
    ]
    dense_matches = _generated_matches_dense(model, scheduler, prompts)
    cache.assert_no_leaks()
    return {
        "name": "recompute_preemption",
        "token_budget": config.max_num_batched_tokens,
        "num_kv_blocks": config.num_kv_blocks,
        "trace": trace,
        "max_scheduled_token_count": max(
            step["scheduled_token_count"] for step in trace
        ),
        "budget_respected": all(
            step["scheduled_token_count"] <= config.max_num_batched_tokens
            for step in trace
        ),
        "preemption_order": preemption_order,
        "completion_order": completion_order,
        "fifo_preserved": completion_order == ["old", "new"],
        "dense_token_match": dense_matches,
        "leak_free": True,
    }


def collect_evidence() -> dict[str, Any]:
    torch.use_deterministic_algorithms(True)
    return {
        "seed": 123,
        "dtype": "float32",
        "kv_block_size": 16,
        "scenarios": [
            _continuous_admission_evidence(),
            _preemption_evidence(),
        ],
    }


if __name__ == "__main__":
    print(json.dumps(collect_evidence(), indent=2, sort_keys=True))
