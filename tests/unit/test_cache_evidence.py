from __future__ import annotations

from bench.cache_correctness import PROMPT_LENGTHS, collect_evidence


def test_cache_correctness_evidence_covers_block_boundaries() -> None:
    evidence = collect_evidence()

    assert tuple(case["prompt_length"] for case in evidence["cases"]) == PROMPT_LENGTHS
    assert all(case["within_internal_tolerance"] for case in evidence["cases"])
    assert all(case["greedy_token_equal"] for case in evidence["cases"])
    assert evidence["canonical_fp32_cache_bytes_for_8_blocks"] == 29_360_128
