---
name: decisions-log
description: How to record design decisions in docs/decisions.md. Use this skill whenever a design fork is discussed or resolved (data structures, block size, eviction policy, scheduler behavior, quantization scheme, API shapes), whenever a benchmark justifies or overturns a choice, and at the end of every milestone, even if the user doesn't ask to document anything.
---

# Decisions Log

Record each design decision in `docs/decisions.md`. Use one entry per decision; do not combine independent choices in one entry.

## Workflow

1. Record the constraint that forced a choice, not merely the implementation task.
2. Compare two or three credible options and state the tradeoff of each.
3. Derive the decision from first principles: memory use, fragmentation, hardware utilization, and latency.
4. Measure the chosen behavior through the appropriate versioned benchmark and add its row to `docs/results.md`.
5. Link the actual result row from the decision entry. A placeholder or unmeasured claim does not complete the entry.
6. Write the interview soundbite **last**, only after the measurement exists. Ground it in a real number and never invent evidence.

Any PR that resolves a design fork without a corresponding `docs/decisions.md` entry is incomplete.

## Entry Template

```markdown
## [Milestone N] Decision title

**Date:** YYYY-MM-DD

**Context:** State the constraint that forced this choice.

**Options considered:**

1. **Option A:** State its tradeoff.
2. **Option B:** State its tradeoff.
3. **Option C:** State its tradeoff, if it is a credible alternative.

**Decision:** State the selected option precisely.

**Why:** Explain the choice from first principles, addressing memory, fragmentation, utilization, and latency where relevant.

**Measured result:** [Results row](results.md#result-anchor) — report the actual metric and benchmark conditions that support or overturn the choice.

**What would change this:** Explain whether the decision changes at 10x model size, 100x traffic, or with a custom CUDA attention backend.

**Interview soundbite:** Give one or two sentences the author can say aloud and defend without notes, grounded in the measured result above.
```

## Reference Examples

Use these as option sets, then replace general tradeoffs with measured project evidence:

- **KV layout:** Block size 16 reduces internal fragmentation but increases block-table entries and lookup overhead; block size 32 reduces table overhead but wastes more tail capacity; contiguous KV is simple and address-efficient but makes growth, reuse, and eviction less flexible and can increase allocation fragmentation or copying.
- **Preemption:** Recompute avoids host-memory use and swap traffic but repeats model work and raises resumed-request latency; swap preserves completed KV work but consumes host memory and PCIe bandwidth, with transfer latency that may dominate short requests.
- **Batching:** Continuous batching admits and retires sequences every step for higher utilization and lower queueing latency, at the cost of scheduler complexity; static batching is simpler and predictable but leaves capacity idle behind uneven sequence lengths and delays new arrivals.
