# mini-vllm Roadmap

mini-vllm is a from-scratch Python and PyTorch inference engine built to connect
model compression with serving efficiency. Its headline result will compare
naive full-precision Transformers generation, mini-vllm FP16 serving, and
mini-vllm INT8 serving using speed, memory, tail latency, and accuracy—not speed
alone.

## Fixed Project Decisions

- Canonical model: `Qwen/Qwen3-0.6B-Base`, not the instruction-tuned checkpoint.
- Canonical accelerator: Colab NVIDIA T4.
- Runtime dtype: FP16 weights, activations, and KV cache on T4; FP32 for portable
  CPU correctness tests. Numerically sensitive reductions may use documented
  FP32 accumulation.
- Package layout: root-level `engine/`, `server/`, and `bench/`; no extra
  `src/mini_vllm/` nesting.
- Configuration: one immutable `EngineConfig` in `engine/config.py`, shared by
  every subsystem. No shadow dictionaries or duplicated architecture constants.
- Core generation, caching, scheduling, sampling, and serving are implemented
  here. `model.generate()`, Transformers generation utilities, vLLM, and TGI
  are banned from the core path.
- Transformers is permitted only as a parity oracle and labeled benchmark
  baseline. The Hugging Face tokenizer is permitted only for encode/decode and
  applicable template handling; safetensors loading is permitted.

## Gate Applied to Every Milestone

**Entry gate:** The previous milestone's required tests are green. The new
design and data structures have been explained, meaningful design forks have
been presented, the user has chosen among them, and the previous milestone's
core-concept quiz has been passed.

**Exit gate:** The deliverable has been demonstrated and all focused, portable,
and applicable parity tests are green. Update `docs/decisions.md` with the
`decisions-log` skill and `docs/results.md` with the `inference-benchmark` skill
before calling the milestone complete. The user receives a design explanation
and core-concept quiz before the next milestone begins.

The user will provide the project skills under `.agents/skills/`. Confirm the
required skill exists and read it before entering its domain; do not create a
replacement when one is missing.

## Milestone 0 — Repository Contract and Roadmap

**Deliverable:** Approve the repository layout, contributor conventions,
hardware target, model target, dtype rules, banned APIs, benchmark discipline,
and single shared-config policy. Create only `AGENTS.md` and `ROADMAP.md`; the
remaining layout is proposed rather than populated.

**Verification:** Confirm that only these two files were added and that both
record `Qwen/Qwen3-0.6B-Base`, the T4 dtype policy, test and parity gates,
benchmark evidence requirements, banned APIs, skill usage, and the shared
`EngineConfig` rule.

**Interview lesson:** Explain why a systems project needs fixed correctness and
measurement contracts before optimization begins.

## Milestone 1 — Correct Forward Pass

**Deliverable:** Load `Qwen/Qwen3-0.6B-Base` weights directly from Hugging Face
safetensors into our own PyTorch implementation. Implement RMSNorm, RoPE,
grouped-query attention with 16 query heads and 8 KV heads, per-head QK-Norm,
SwiGLU, 28 decoder layers, hidden size 1024, head dimension 128, and tied token
embeddings. The public result is prompt tokens in and logits out; greedy decode
logic is ours.

**Verification:** Use the `hf-parity-testing` skill. Compare intermediate layer
outputs to the Transformers implementation to localize divergence, then require
token-for-token greedy equality on five fixed prompts. All correctness tests
must run on CPU in FP32; T4 FP16 parity is an additional target.

**Interview lesson:** Defend GQA's tensor shapes and KV-memory savings; explain
why QK-Norm, RoPE conventions, weight mapping, and tied embeddings are common
sources of silent model drift.

## Milestone 2 — PagedAttention KV Cache

**Deliverable:** Add fixed-size physical KV blocks, initially 16 tokens per
block; an allocator with allocate/free/refcount operations; and a logical-to-
physical block table per sequence. Implement attention in pure PyTorch by
gathering K/V through block tables. Mark the precise paged K/V input-output
boundary where the future `apex-attention` CUDA kernel will plug in.

**Verification:** Test exhaustion, allocation, free, refcount behavior,
double-free rejection, leak detection, physical-block reuse after sequence
retirement, partially filled final blocks, and independent block tables. Require
logit/output parity with the uncached Milestone 1 path across block boundaries.

**Interview lesson:** Explain how paging avoids per-sequence contiguous
reservations, reduces external fragmentation, and enables non-contiguous growth
while distinguishing that benefit from unavoidable internal fragmentation in a
partially filled final block.

## Milestone 3 — Continuous Batching Scheduler

**Deliverable:** Implement a step loop that admits requests every iteration,
batches active sequences for one decode step, retires finished sequences
mid-batch, and enforces a maximum-token budget. Before coding, choose and record
the preemption/eviction policy and the state needed to resume work.

**Verification:** Test admission and retirement invariants, token-budget
enforcement, progress under saturation, no starvation, prompt-length diversity,
block release on success and failure, and fair treatment of short requests when
a long request is active.

**Interview lesson:** Contrast static and continuous batching, then explain how
iteration-level admission and retirement reduce GPU bubbles and improve useful
batch occupancy.

## Milestone 4 — OpenAI-Compatible Server

**Deliverable:** Add FastAPI `POST /v1/completions` with non-streaming and SSE
streaming responses. Implement request validation and our own greedy,
temperature, and top-p sampling. Match the OpenAI completion response closely
enough for a standard client to work without modification.

**Verification:** Test response schemas, finish reasons, usage counts,
deterministic greedy behavior, seeded sampling where supported, top-p and
temperature edge cases, SSE framing and terminal event, disconnect cleanup,
invalid requests, and concurrent client requests.

**Interview lesson:** Explain the separation among transport, scheduler,
sampling, and model execution, including how streaming changes cleanup and
latency semantics without changing model math.

## Milestone 5 — Reproducible Benchmarks

**Deliverable:** Using the `inference-benchmark` skill, build a versioned harness
that compares mini-vllm with vanilla Transformers `model.generate()` at
concurrency 1, 8, and 32. Measure aggregate tokens/second, time to first token,
p50/p99 end-to-end latency, and peak memory. Record T4 hardware, software
versions, dtypes, prompt/output distributions, scheduler limits, synchronization,
warmups, and raw samples.

**Verification:** Test metric calculations with synthetic timings, ensure both
systems receive equivalent tokenized workloads and stopping rules, isolate
warmups from measured runs, synchronize CUDA at timing boundaries, and confirm
peak-memory counters are reset consistently. Publish the resulting Markdown
table to `docs/results.md` only through the harness.

**Interview lesson:** Defend benchmark fairness, distinguish throughput from
TTFT and tail latency, and explain why concurrency changes the system limiter.

## Milestone 6 — Quantized Serving

**Deliverable:** Using the `quantized-serving` skill, implement INT8 weight-only
quantization with per-output-channel scales as the first compressed backend.
Document the scaling equation, which modules are quantized or excluded, whether
calibration data is needed for the chosen weight-only method, and the runtime
dequantization/compute path. Serve the quantized model through the same cache,
scheduler, sampling, and server interfaces.

Publish one headline comparison:

| System | Weight dtype | Tokens/s | Peak VRAM | TTFT | p50 | p99 | Perplexity | Task quality | Drift notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Naive Transformers generation | FP16 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Reference |
| mini-vllm | FP16 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Serving-only delta |
| mini-vllm | INT8 weight-only | TBD | TBD | TBD | TBD | TBD | TBD | TBD | Quantization delta |

**Verification:** Compare FP16 and INT8 logits on fixed inputs, perplexity on a
fixed versioned evaluation set, a small deterministic task-quality check, and
end-to-end serving behavior. Run the same benchmark workloads for all three
systems and record parity drift rather than hiding it.

**Interview lesson:** Explain per-channel versus per-tensor scaling, symmetric
versus asymmetric quantization, weight-only bandwidth savings versus
dequantization cost, why kernel support determines realized speedup, and the
accuracy/memory implications of quantizing the KV cache next.

## Milestone 7 — Pluggable Attention Backend (Stretch)

**Deliverable:** Define the smallest attention-backend interface that accepts
paged K/V state plus query and sequence metadata and returns attention output.
Register the pure-PyTorch implementation as backend one and provide a clean
registration point for an `apex-attention` CUDA backend without changing model,
cache-manager, or scheduler semantics.

**Verification:** Run identical shape, mask, multi-sequence, block-boundary, and
numerical parity tests against every registered backend. Verify that backend
selection occurs through the shared `EngineConfig` and requires no scheduler
branching.

**Interview lesson:** Defend the minimal backend contract and explain how a
stable systems boundary lets kernels evolve independently from request
scheduling and cache ownership.

## Planned Documentation Story

After implementation begins, `docs/decisions.md` will record each milestone's
alternatives, choice, evidence, scale implications, and interview soundbite.
`docs/results.md` will contain only harness-produced measurements with complete
conditions.

The final README will be drafted as a case study: two short paragraphs connect
measured go-kart optimization to measured inference optimization, followed by
an architecture diagram, the headline speed-and-accuracy table, and exact run
instructions. The user will rewrite that draft in their own voice.
