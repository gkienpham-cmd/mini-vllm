from __future__ import annotations

from dataclasses import replace
from unittest.mock import patch

import pytest
import torch

from engine.cache import BlockAllocator, CacheExhaustedError, PagedKVCache
from engine.model.qwen3 import Qwen3ForCausalLM


def test_allocator_exhaustion_refcounts_double_free_and_leaks() -> None:
    allocator = BlockAllocator(2)

    first = allocator.allocate()
    second = allocator.allocate()
    assert (first, second) == (0, 1)
    with pytest.raises(CacheExhaustedError, match="only 0 free"):
        allocator.allocate()

    allocator.incref(first)
    assert allocator.refcount(first) == 2
    allocator.free(first)
    assert allocator.refcount(first) == 1
    with pytest.raises(RuntimeError, match="leaked KV blocks"):
        allocator.assert_no_leaks()

    allocator.free(first)
    with pytest.raises(RuntimeError, match="not allocated"):
        allocator.free(first)
    allocator.free(second)
    allocator.assert_no_leaks()


def test_allocator_reuses_most_recently_freed_block() -> None:
    allocator = BlockAllocator(3)
    blocks = allocator.allocate_many(3)
    allocator.free(blocks[0])
    allocator.free(blocks[2])

    assert allocator.allocate() == blocks[2]


def test_sequence_tables_are_independent_and_partial_blocks_are_reused(
    tiny_config,
) -> None:
    config = replace(tiny_config, num_kv_blocks=3)
    cache = PagedKVCache(config)
    cache.create_sequence("first")
    cache.create_sequence("second")

    first_append = cache.begin_append(["first"], query_length=17)
    cache.commit(first_append)
    second_append = cache.begin_append(["second"], query_length=1)
    cache.commit(second_append)

    assert cache.sequence_state("first").block_ids == [0, 1]
    assert cache.sequence_state("first").num_tokens == 17
    assert cache.sequence_state("second").block_ids == [2]
    assert cache.sequence_state("second").num_tokens == 1

    with pytest.raises(CacheExhaustedError):
        cache.begin_append(["second"], query_length=16)
    assert cache.sequence_state("second").block_ids == [2]
    assert cache.sequence_state("second").num_tokens == 1

    cache.release_sequence("first")
    growing_second = cache.begin_append(["second"], query_length=16)
    cache.commit(growing_second)
    assert cache.sequence_state("second").block_ids == [2, 0]

    cache.release_sequence("second")
    cache.assert_no_leaks()


def test_append_rollback_is_atomic_across_sequences(tiny_config) -> None:
    config = replace(tiny_config, num_kv_blocks=4)
    cache = PagedKVCache(config)
    cache.create_sequence("a")
    cache.create_sequence("b")

    reservation = cache.begin_append(["a", "b"], query_length=17)
    assert cache.allocator.free_count == 0
    cache.rollback(reservation)

    assert cache.sequence_state("a").block_ids == []
    assert cache.sequence_state("b").block_ids == []
    assert cache.sequence_state("a").num_tokens == 0
    assert cache.sequence_state("b").num_tokens == 0
    assert cache.allocator.free_count == 4
    with pytest.raises(RuntimeError, match="rolled_back"):
        cache.commit(reservation)

    cache.release_sequence("a")
    cache.release_sequence("b")
    cache.assert_no_leaks()


def test_cached_model_matches_dense_prefill_and_decode_across_block_boundary(
    tiny_config,
) -> None:
    config = replace(tiny_config, num_kv_blocks=4)
    model = Qwen3ForCausalLM(config).eval()
    cache = PagedKVCache(config)
    cache.create_sequence("request")
    prompt = torch.arange(17).remainder(config.vocab_size)[None, :]

    expected_prompt = model(prompt).logits
    actual_prompt = model.forward_cached(
        prompt, cache=cache, sequence_ids=["request"]
    ).logits
    torch.testing.assert_close(actual_prompt, expected_prompt, rtol=1e-5, atol=1e-6)

    next_input = torch.tensor([[7]])
    expected_decode = model(torch.cat((prompt, next_input), dim=1)).logits[:, -1]
    actual_decode = model.forward_cached(
        next_input, cache=cache, sequence_ids=["request"]
    ).logits[:, -1]
    # One-token and full-prefix GEMMs have slightly different FP32 reduction order.
    torch.testing.assert_close(actual_decode, expected_decode, rtol=1e-5, atol=2e-6)

    cache.release_sequence("request")
    cache.assert_no_leaks()


def test_batched_decode_supports_independent_context_lengths(tiny_config) -> None:
    config = replace(tiny_config, num_kv_blocks=6)
    model = Qwen3ForCausalLM(config).eval()
    cache = PagedKVCache(config)
    cache.create_sequence("long")
    cache.create_sequence("short")
    long_prompt = torch.arange(17).remainder(config.vocab_size)[None, :]
    short_prompt = torch.tensor([[3]])
    model.forward_cached(long_prompt, cache=cache, sequence_ids=["long"])
    model.forward_cached(short_prompt, cache=cache, sequence_ids=["short"])

    next_inputs = torch.tensor([[7], [8]])
    actual = model.forward_cached(
        next_inputs, cache=cache, sequence_ids=["long", "short"]
    ).logits[:, -1]
    expected = torch.cat(
        (
            model(torch.cat((long_prompt, next_inputs[0:1]), dim=1)).logits[:, -1],
            model(torch.cat((short_prompt, next_inputs[1:2]), dim=1)).logits[:, -1],
        ),
        dim=0,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)

    cache.release_sequence("long")
    cache.release_sequence("short")
    cache.assert_no_leaks()


def test_model_failure_rolls_back_new_blocks_and_length(tiny_config) -> None:
    config = replace(tiny_config, num_kv_blocks=2)
    model = Qwen3ForCausalLM(config).eval()
    cache = PagedKVCache(config)
    cache.create_sequence("request")
    prompt = torch.arange(17).remainder(config.vocab_size)[None, :]

    with patch.object(
        model.model.layers[1],
        "forward_cached",
        side_effect=RuntimeError("injected layer failure"),
    ):
        with pytest.raises(RuntimeError, match="injected layer failure"):
            model.forward_cached(prompt, cache=cache, sequence_ids=["request"])

    assert cache.sequence_state("request").block_ids == []
    assert cache.sequence_state("request").num_tokens == 0
    assert cache.allocator.free_count == 2

    cache.release_sequence("request")
    cache.assert_no_leaks()
