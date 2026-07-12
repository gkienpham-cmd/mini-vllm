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

## [Milestone 2] Configure cache capacity in physical blocks

**Date:** 2026-07-13

**Context:** Cache exhaustion and allocated memory must be explicit without
introducing rounding policy between a token or byte budget and physical blocks.

**Options considered:**

1. **Physical-block count:** Maps one configuration value directly to allocator
   capacity and tensor allocation, but callers must understand block size.
2. **Token capacity:** Is friendlier at the request boundary but must define how
   non-multiples of the block size round.
3. **Memory budget:** Adapts to model shape and dtype but adds derivation policy
   before the scheduler needs it.

**Decision:** Add immutable `kv_block_size=16` and `num_kv_blocks` runtime
choices to `EngineConfig`. Zero keeps dense-only loading available; cache
construction requires a positive block count.

**Why:** Physical blocks are the allocator's real unit, so exhaustion is exact
and no hidden rounding can allocate more memory than configured. Sixteen-token
blocks preserve the roadmap's balance between tail waste and block-table size.

**Measured result:** [Milestone 2 CPU FP32 paged-cache correctness](results.md#milestone-2-cpu-fp32-paged-cache-correctness-2026-07-13)
— 8 blocks for the canonical FP32 model allocate exactly 28 MiB; boundary tests
covered the empty, partial, full, and second-block states.

**What would change this:** At 10x model size, a byte-budget helper may derive
the block count before constructing `EngineConfig`. At 100x traffic, admission
still consumes physical blocks. A CUDA backend may constrain supported block
sizes but should not change capacity ownership.

**Interview soundbite:** I configured the resource the allocator actually owns:
8 canonical FP32 blocks are exactly 28 MiB, with no token-to-block rounding
hidden inside allocation.

## [Milestone 2] Keep readable sequence tables and tensorize at the backend boundary

**Date:** 2026-07-13

**Context:** Sequence growth and retirement need inspectable metadata, while an
attention kernel eventually needs dense tensor inputs.

**Options considered:**

1. **Python block-ID lists:** Make lifecycle transitions explicit and allocate
   metadata only as sequences grow, but require materialization before attention.
2. **Fixed tensors:** Provide stable backend shapes but reserve maximum table
   capacity and require sentinel bookkeeping for every sequence.
3. **Dynamic tensors:** Match the backend type directly but reallocate metadata
   on append.

**Decision:** Store each sequence's physical block IDs in a Python list with an
explicit committed token count. Materialize a padded block-table tensor plus
context lengths for each append batch.

**Why:** Scheduler-visible state remains readable and independent per sequence,
while the pure-PyTorch/backend boundary sees only tensors. Padding exists only
for the active batch and never changes ownership semantics.

**Measured result:** [Milestone 2 CPU FP32 paged-cache correctness](results.md#milestone-2-cpu-fp32-paged-cache-correctness-2026-07-13)
— independent 17-token and 1-token tables were batched through one decode step,
and the five full-checkpoint prompts matched Hugging Face exactly.

**What would change this:** At 100x traffic, profiling may justify persistent
device-side tables to reduce host-to-device metadata work. A CUDA backend can
consume the materialized tensors without changing scheduler or cache ownership.

**Interview soundbite:** I kept ownership as readable lists but tensorized only
at the kernel boundary; variable-context batching passed and all five paged
prompt outputs matched the reference token-for-token.

## [Milestone 2] Preserve dense forward as a separate correctness path

**Date:** 2026-07-13

**Context:** Cache integration must not weaken the known-correct Milestone 1
oracle or make every dense attention call branch on optional cache state.

**Options considered:**

1. **Separate cached method:** Shares projections and weights while keeping
   dense `forward` semantics isolated, at the cost of a second traversal method.
2. **Optional cache argument:** Reduces public methods but branches through the
   oracle path and expands its state space.
3. **Full backend abstraction now:** Establishes the stretch boundary early but
   adds registration machinery before a second backend exists.

**Decision:** Keep dense `forward` unchanged and add explicit `forward_cached`
methods, sharing only Q/K/V projection and output-layout helpers.

**Why:** The dense path stays simple enough to diagnose cache drift, while the
cached path makes reservation and ownership explicit. The pure-PyTorch paged
function marks the future backend boundary without pulling Milestone 7 forward.

**Measured result:** [Milestone 2 CPU FP32 paged-cache correctness](results.md#milestone-2-cpu-fp32-paged-cache-correctness-2026-07-13)
— the portable suite passed 26 tests and the parity suite passed all 8 CPU
tests, including both dense and paged exact-token checks.

**What would change this:** More attention backends should share the paged
function contract, not merge cache branching into dense forward. Model scale or
traffic does not change the need for a simple oracle.

**Interview soundbite:** I paid for a second explicit traversal to keep the
oracle trustworthy; 8 CPU parity tests passed with dense and paged generation
checked independently.

## [Milestone 2] Use a refcounted LIFO free list

**Date:** 2026-07-13

**Context:** Allocation and retirement happen on every sequence lifecycle and
must reject double frees while making reuse deterministic.

**Options considered:**

1. **LIFO free list:** Provides O(1) allocation/free and immediate reuse of the
   most recently returned block, without sorted-order guarantees.
2. **Lowest-ID heap:** Makes the chosen ID predictable but costs O(log n) per
   operation without improving memory capacity.
3. **Bitmap scan:** Uses compact metadata but makes allocation O(number of
   blocks) in the worst case.

**Decision:** Use a LIFO free list plus one integer refcount per physical block.
Initial allocation remains 0, 1, ... for readable tests.

**Why:** Block identity has no semantic value. Constant-time lifecycle work and
recent-block reuse matter more than selecting the numerically smallest ID;
refcounts leave room for future shared prefixes without permitting double free.

**Measured result:** [Milestone 2 CPU FP32 paged-cache correctness](results.md#milestone-2-cpu-fp32-paged-cache-correctness-2026-07-13)
— the deterministic allocator trace exhausted capacity, exercised refcount 2,
rejected double free, reused the most recently freed block, and ended leak-free.

**What would change this:** At 100x traffic, allocator synchronization and
per-device sharding become the next concern. A CUDA backend does not need to
know how the scheduler selected physical block IDs.

**Interview soundbite:** The allocator trace covered exhaustion, refcount 2,
double-free rejection, LIFO reuse, and a zero-leak finish with constant-time
free-list operations.

## [Milestone 2] Prefill prompts through paged attention

**Date:** 2026-07-13

**Context:** The paged path must prove correctness across block boundaries and
must not make prompt processing one model invocation per token.

**Options considered:**

1. **Whole-prompt paged prefill:** Projects the prompt once, writes K/V to
   blocks, and exercises the same gather boundary as decode.
2. **Token-by-token prefill:** Reuses one append shape but performs needless
   serial model work.
3. **Dense prefill then populate:** Reuses dense attention but leaves prompt
   attention untested through block tables.

**Decision:** Append and attend to a uniform-length prompt batch through the
pure-PyTorch paged boundary, then use one-token appends for decode.

**Why:** It avoids intentional serial prefill and validates block-table gathers
for multi-token queries. Causal masking uses absolute query positions, so the
same boundary handles both prefill and decode.

**Measured result:** [Milestone 2 CPU FP32 paged-cache correctness](results.md#milestone-2-cpu-fp32-paged-cache-correctness-2026-07-13)
— lengths 15, 16, and 17 covered both sides of the boundary; all five canonical
prompts matched Hugging Face token-for-token.

**What would change this:** At larger scale, chunked prefill may cap temporary
attention memory. At 100x traffic, the scheduler may batch compatible chunks.
A CUDA backend can replace the gather/attention function with the same metadata.

**Interview soundbite:** Prompt lengths 15, 16, and 17 exercised both sides of
the 16-token boundary, and whole-prompt paged prefill still matched every
canonical generated token.

## [Milestone 2] Make cache appends transactional

**Date:** 2026-07-13

**Context:** Exhaustion or a model failure must not leave token counts advanced,
partially extended block tables, or leaked blocks for the scheduler to guess at.

**Options considered:**

1. **Atomic rollback:** Reserve every block first and commit lengths only after
   logits succeed, with extra reservation bookkeeping.
2. **Release affected sequences:** Avoids partial state but discards valid prior
   K/V work after a transient failure.
3. **Caller cleanup:** Minimizes cache code but exposes partial mutation and
   duplicates recovery logic in every caller.

**Decision:** Reserve the whole append batch before writes, track new blocks,
commit only after the LM head succeeds, and roll back newly reserved blocks on
any exception. Writes beyond committed length are ignored and overwritten later.

**Why:** The committed token count is the visibility boundary. This preserves
prior K/V work, makes failure semantics local to the cache/model contract, and
prevents scheduler-level leak recovery from becoming normal control flow.

**Measured result:** [Milestone 2 CPU FP32 paged-cache correctness](results.md#milestone-2-cpu-fp32-paged-cache-correctness-2026-07-13)
— multi-sequence exhaustion left both tables unchanged, and an injected layer
failure returned both blocks while restoring the sequence length to zero.

**What would change this:** At 100x traffic, reservations may need locks or a
single scheduler owner, but the commit/rollback semantics remain. CUDA failures
must propagate through the same transaction boundary.

**Interview soundbite:** An injected mid-model failure returned both reserved
blocks and restored length zero, so the scheduler never has to interpret a
half-committed cache append.

## [Milestone 2] Bound dense-versus-cached FP32 drift separately from HF parity

**Date:** 2026-07-13

**Context:** Dense full-prefix recomputation and one-token cached decode use
different GEMM shapes, so their FP32 reduction order can differ even when both
match the Hugging Face oracle.

**Options considered:**

1. **Separate internal tolerance:** Keep direct HF parity at the strict default
   and allow only the local dense-versus-cached comparison a measured bound.
2. **Force identical projection shapes:** Process dense tokens serially to align
   GEMMs, sacrificing the readable dense implementation and prefill behavior.
3. **Use token equality only:** Avoid numerical bounds but lose an early warning
   for logit drift that has not yet changed argmax.

**Decision:** Keep direct Hugging Face CPU parity at `rtol=1e-5`, `atol=1e-6`.
Use `rtol=1e-5`, `atol=2e-6` only for dense full-prefix versus cached one-token
logits, while still requiring exact greedy tokens.

**Why:** The tighter HF oracle continues to catch model errors. The separate
local bound measures the expected GEMM-shape effect without serializing the
dense path or reducing the assertion to token IDs.

**Measured result:** [Milestone 2 CPU FP32 paged-cache correctness](results.md#milestone-2-cpu-fp32-paged-cache-correctness-2026-07-13)
— the versioned five-length harness observed maximum absolute error
5.7220458984375e-6, all cases passed the combined relative/absolute bound, and
all greedy tokens agreed. Direct HF parity passed at the stricter default.

**What would change this:** A larger measured error, token divergence, or HF
failure invalidates the bound. CUDA FP16 keeps its existing independent
`rtol=1e-3`, `atol=1e-3` parity policy.

**Interview soundbite:** I isolated a measured GEMM-shape effect instead of
weakening the oracle: HF stayed at `1e-5/1e-6`, while the local cached comparison
used `1e-5/2e-6` and preserved exact greedy tokens.
