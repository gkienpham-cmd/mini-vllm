# Design decisions

## [Milestone 1] Construct the 0.6B model normally on CPU

**Date:** 2026-07-12

**Context:** The loader must be easy to inspect and must expose missing, extra,
or wrongly shaped checkpoint tensors. Lower peak host memory matters, but the
canonical 0.6B model fits the development machine and Colab host memory.

**Options considered:**

1. **Ordinary CPU construction:** Keeps module initialization, parameter names,
   and error handling straightforward, at the cost of allocating parameters
   before checkpoint values are copied in.
2. **Meta-device construction:** Avoids initialized parameter storage but makes
   materialization and incomplete-load detection more subtle.
3. **Layer-at-a-time construction and loading:** Minimizes peak memory further
   but couples module construction to checkpoint traversal.

**Decision:** Construct the full model normally in FP32 on CPU, copy one
safetensors tensor at a time with exact key and shape checks, then move the
loaded model to its configured device and dtype.

**Why:** Milestone 1 optimizes for a loader whose failure modes are visible. The
model and checkpoint use the same `[out_features, in_features]` linear layout,
so the loader rejects shape differences and never guesses a transpose. Reading
each safetensors shard once avoids a second full checkpoint dictionary while
preserving the ordinary, debuggable module lifecycle.

**Measured result:** [Milestone 1 CPU FP32 correctness](results.md#milestone-1-cpu-fp32-correctness-2026-07-12)
— all 310 stored checkpoint tensors were consumed, the omitted tied LM-head
alias was handled explicitly, and full-checkpoint parity passed.

**What would change this:** At 10x model size, ordinary construction would put
avoidable pressure on host memory. Switch the construction boundary to meta
tensors or layer-at-a-time materialization while retaining the same strict load
report. Traffic volume does not change weight-loading semantics.

**Interview soundbite:** For the 0.6B reference I spent memory to buy
inspectability: the loader accounted for all 310 stored tensors and passed full
CPU parity. The loader boundary is narrow enough to replace with meta-device
materialization when model size, rather than correctness debugging, becomes the
limiter.

## [Milestone 1] Precompute the fixed-context RoPE tables

**Date:** 2026-07-12

**Context:** Qwen3-0.6B-Base has a fixed 32,768-token context and no RoPE
scaling. Greedy decoding in Milestone 1 recomputes the whole prefix, so RoPE
must stay simple and deterministic while matching Hugging Face's split-half
rotation convention.

**Options considered:**

1. **Precompute the full table:** Pays fixed storage once and makes every
   forward pass an indexed lookup.
2. **Grow a cache on demand:** Stores only reached positions but introduces
   mutable state, growth policy, and reallocation behavior.
3. **Recompute each forward:** Avoids cache state but repeats trigonometric work
   at every greedy step.

**Decision:** Precompute cosine and sine tables in FP32 for the configured
context, register them as non-persistent buffers, and cast only the selected
rows to the activation dtype.

**Why:** A fixed table removes cache lifecycle decisions from the correctness
milestone. FP32 construction matches the reference's numerically stable angle
calculation; non-persistent buffers move with the model without polluting the
checkpoint key space.

**Measured result:** [Milestone 1 CPU FP32 correctness](results.md#milestone-1-cpu-fp32-correctness-2026-07-12)
— direct RoPE tensor parity, the tiny full model, all captured full-checkpoint
layer boundaries, and all five greedy prompt cases passed at the declared FP32
tolerance.

**What would change this:** A much longer or dynamically scaled context would
make an on-demand cache more attractive. Model size and traffic do not directly
change the table, while a custom attention kernel may prefer a different RoPE
input contract and would need backend parity tests.

**Interview soundbite:** Because the checkpoint has a fixed 32K context, I
precomputed RoPE in FP32 and reduced runtime behavior to an indexed lookup. It
matched Hugging Face at the component, layer-boundary, and five-prompt greedy
levels.

## [Milestone 1] Use strict two-tier parity testing

**Date:** 2026-07-12

**Context:** Full-checkpoint greedy parity is authoritative but expensive and
does not localize mathematical errors. Tiny tests are fast and diagnostic but
cannot prove that the real safetensors mapping and full 28-layer model agree
with Hugging Face.

**Options considered:**

1. **Strict two-tier suite:** Use deterministic tiny module/model tests for
   diagnosis and require the real checkpoint for the parity merge gate.
2. **Opt-in full-checkpoint parity:** Keeps default test runs lighter but allows
   changes to merge without exercising the canonical model.
3. **Full-checkpoint tests only:** Exercises real weights but makes failures
   slow and difficult to bisect.

**Decision:** Keep focused deterministic component tests and a mandatory
`tests/parity` suite containing the canonical full-checkpoint test. CPU FP32 is
the portable oracle; the T4 FP16 case is an additional CUDA-marked gate.

**Why:** The first tier identifies whether RMSNorm, RoPE, SwiGLU, attention, or
a layer boundary diverged. The second tier catches checkpoint mapping and error
accumulation across all 28 layers. Exact generated token IDs are checked at
every step rather than comparing decoded strings.

**Measured result:** [Milestone 1 CPU FP32 correctness](results.md#milestone-1-cpu-fp32-correctness-2026-07-12)
— 14 focused tests passed, the canonical CPU suite passed 15 tests, and the
parity gate passed all seven CPU parity tests across exactly five fixed prompts.

**What would change this:** At 100x traffic, test structure stays the same but
scheduler/load tests become additional gates. At 10x model size, full parity may
move to dedicated hardware, but it must remain required rather than optional.
Every custom attention backend must pass the same reference suite.

**Interview soundbite:** Tiny parity tests tell me where the math first breaks;
the full checkpoint proves the assembled engine computes the intended model.
Both tiers passed, including exact token checks on the five fixed prompts.

