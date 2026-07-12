# Measured results

Performance results belong here only after they are produced by a versioned
benchmark harness. Milestone 1 reports correctness evidence only; it makes no
throughput, latency, or memory-performance claim.

## Milestone 1 CPU FP32 correctness — 2026-07-12

**What changed:** Added the from-scratch dense Qwen3 forward pass, strict
safetensors loader, hidden-state diagnostics, and greedy decoding without
Transformers generation utilities.

**Environment:**

- Hardware: Apple M1 Pro MacBook Pro, 16 GB RAM, CPU execution.
- Python 3.12.4; PyTorch 2.13.0; Transformers 4.57.6.
- safetensors 0.8.0; huggingface-hub 0.36.2; pytest 9.1.1.
- Checkpoint: `Qwen/Qwen3-0.6B-Base` revision
  `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`.
- mini-vllm dtype: FP32; Hugging Face oracle dtype: FP32.
- Deterministic Python/PyTorch seeds; deterministic PyTorch algorithms enabled.
- CPU tensor tolerance: `rtol=1e-5`, `atol=1e-6`.
- Greedy decode: exactly five fixed prompts and at most eight new tokens each;
  identical tokenizer, input IDs, EOS rule, and token budget.

| Correctness check | Result |
| --- | --- |
| RMSNorm tensor parity | Pass |
| RoPE tensor parity | Pass |
| SwiGLU tensor parity | Pass |
| GQA + per-head QK-Norm attention parity | Pass |
| Tiny two-layer logits and hidden-state parity | Pass |
| Checkpoint coverage | Pass: all 310 stored tensors consumed |
| Tied embedding/LM-head storage | Pass: one shared parameter; omitted alias recognized |
| Full 28-layer hidden-state diagnostics | Pass: no divergent captured boundary |
| Five-prompt greedy token parity | Pass: every generated token position matched |
| Portable command: `python -m pytest -q -m "not cuda and not benchmark"` | 15 passed, 1 deselected |
| Parity command: `python -m pytest -q tests/parity` | 7 passed, 1 CUDA test skipped |
| T4 FP16 full-checkpoint parity | Not run: no CUDA device on this machine |

No benchmark row is published for this milestone. Concurrency, TTFT,
throughput, p50/p99 latency, and peak VRAM measurements begin only when the
versioned benchmark harness can run equivalent workloads.

## Milestone 2 CPU FP32 paged-cache correctness — 2026-07-13

**What changed:** Added 16-token physical KV blocks, a refcounted allocator,
per-sequence block tables, transactional cache appends, whole-prompt paged
prefill, one-token cached decode, and a pure-PyTorch paged-attention boundary.
The dense Milestone 1 path remains unchanged as the local oracle.

**Environment and cache conditions:**

- Same Apple M1 Pro CPU, Python 3.12.4, PyTorch 2.13.0, Transformers 4.57.6,
  and canonical checkpoint revision recorded for Milestone 1.
- CPU FP32 model and KV cache; deterministic seed 123 and deterministic PyTorch
  algorithms.
- Block size 16. Full-checkpoint parity used 8 physical blocks. Each layer owns
  separate K and V tensors shaped `[8, 16, 8, 128]`, for exactly 29,360,128
  bytes (28 MiB) across all 28 layers.
- The versioned `python -m bench.cache_correctness` harness covered prompt
  lengths 1, 15, 16, 17, and 31 with one cached decode token.

| Correctness check | Result |
| --- | --- |
| Allocation, exhaustion, refcount, and double-free behavior | Pass |
| Leak detection and LIFO physical-block reuse | Pass |
| Independent sequence tables and partially filled final blocks | Pass |
| Multi-sequence append rollback on exhaustion | Pass: tables, lengths, and free count remained unchanged |
| Injected model-layer failure | Pass: two reserved blocks returned; token count reset to 0 |
| Variable-context batched decode | Pass across 17-token and 1-token contexts |
| Dense versus paged tiny-model logits | Pass at `rtol=1e-5`, `atol=2e-6`; maximum absolute error 5.7220458984375e-6 across the five boundary cases |
| Direct Hugging Face tiny-model paged parity | Pass at `rtol=1e-5`, `atol=1e-6` |
| Five-prompt full-checkpoint paged greedy parity | Pass: every generated token position matched Hugging Face exactly |
| Portable command: `python -m pytest -q -m "not cuda and not benchmark"` | 26 passed, 1 deselected |
| Parity command: `python -m pytest -q tests/parity` | 8 passed, 1 CUDA test skipped |
| T4 FP16 paged-cache parity | Not run: no CUDA device on this machine |

The internal dense-versus-paged tolerance changes only the local recomputation
comparison. The direct Hugging Face CPU oracle retains the stricter project
default. The small drift comes from comparing a one-token cached GEMM with a
full-prefix dense GEMM, whose FP32 reductions use different matrix shapes.

No throughput, latency, or memory-performance claim is published for this
milestone. The 28 MiB figure is exact allocated tensor capacity, not measured
peak process memory. Comparable concurrency, TTFT, p50/p99, throughput, and
peak-memory measurements remain deferred to the versioned Milestone 5 harness.

## Milestone 3 CPU FP32 scheduler correctness — 2026-07-13

**What changed:** Added a decode-first continuous batching scheduler with a hard
per-step token budget, chunked prompt/recompute prefill, original-arrival FIFO
admission, immediate mid-batch retirement, and recompute preemption under KV
pressure.

**Environment and scheduler conditions:**

- Same Apple M1 Pro CPU, Python 3.12.4, PyTorch 2.13.0, Transformers 4.57.6,
  deterministic seed 123, and CPU FP32 policy recorded for prior milestones.
- Block size 16; greedy selection only. Every model input token, including
  prompt and recompute tokens, counts against `max_num_batched_tokens`.
- The versioned `python -m bench.scheduler_correctness` harness records every
  step's scheduled-token count, admissions, emissions, preemptions, request
  states, and free-block count. It records no wall-clock measurements.

| Correctness scenario | Continuous admission | Recompute saturation |
| --- | --- | --- |
| Prompt lengths | Long 3; short 1 | Old 16; new 16 |
| Output-token limits | Long 6; short 1 | Old 20; new 4 |
| Physical blocks / token budget | 6 / 4 | 3 / 2 |
| Maximum scheduled tokens in one step | 3 of 4 | 2 of 2 |
| Preemption | None | New request once |
| Completion evidence | Short step 2; long step 6 | Old step 27; new step 37 |
| Original FIFO age preserved | Pass | Pass: old then new |
| Dense greedy token equality | Pass for both requests | Pass for both requests |
| Final cache state | Pass: no live sequences or blocks | Pass: no live sequences or blocks |

Additional verification:

| Correctness check | Result |
| --- | --- |
| Chunked prompt and recompute budget enforcement | Pass: no trace exceeded its configured token budget |
| Repeated arrivals | Pass: the oldest request advanced on every decode iteration and finished |
| Mid-batch admission and retirement | Pass: the short request joined and retired while the long request remained active |
| Prefill failure cleanup | Pass: request marked failed and all blocks released |
| Batched-decode failure cleanup | Pass: affected requests marked failed, unrelated state preserved, and failure diagnostics propagated |
| Five-prompt scheduler token parity | Pass: every generated token position matched Hugging Face exactly |
| Full command: `python -m pytest -q` | 37 passed, 1 CUDA test skipped |
| Portable command: `python -m pytest -q -m "not cuda and not benchmark"` | 37 passed, 1 deselected |
| Parity command: `python -m pytest -q tests/parity` | 8 passed, 1 CUDA test skipped |
| T4 FP16 scheduler parity | Not run: no CUDA device on this machine |

This is correctness and scheduling-policy evidence, not a performance result.
No throughput, TTFT, latency percentile, or peak-memory claim is published;
equivalent Transformers and mini-vllm measurements remain deferred to the
versioned Milestone 5 benchmark harness.
