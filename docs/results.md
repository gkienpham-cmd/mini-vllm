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
