from __future__ import annotations

from bench.scheduler_correctness import collect_evidence


def test_scheduler_correctness_evidence_records_raw_invariants() -> None:
    evidence = collect_evidence()
    continuous, preemption = evidence["scenarios"]

    assert continuous["name"] == "continuous_admission"
    assert continuous["budget_respected"]
    assert continuous["short_finished_while_long_active"]
    assert all(continuous["dense_token_match"].values())
    assert continuous["leak_free"]
    assert continuous["trace"]

    assert preemption["name"] == "recompute_preemption"
    assert preemption["budget_respected"]
    assert preemption["preemption_order"] == ["new"]
    assert preemption["fifo_preserved"]
    assert all(preemption["dense_token_match"].values())
    assert preemption["leak_free"]
    assert preemption["trace"]
