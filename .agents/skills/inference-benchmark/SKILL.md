---
name: inference-benchmark
description: How to benchmark inference performance honestly and update docs/results.md. Use this skill whenever measuring speed, memory, latency, or throughput, whenever the user claims or asks about a speedup, after completing any milestone, and before writing any performance number into a README, results table, or commit message, even if the user doesn't say "benchmark".
---

# Inference Benchmark

## 1. Run the Fixed Protocol

- Run every measurement through a versioned script under `bench/`.
- Run at least three warmup iterations and discard every warmup result.
- Run at least five timed iterations and report their median.
- Call `torch.cuda.synchronize()` immediately before starting and immediately after stopping every GPU timer.
- Reuse the same fixed prompt set and the same fixed `max_new_tokens` for each scenario so results remain comparable across weeks.
- Preserve every raw timed measurement in the harness output; never retain only aggregates.

## 2. Measure the Complete Metric Set

- Report all metrics for every variant at concurrency 1, 8, and 32.
- Report aggregate tokens/second and per-request tokens/second.
- Report time to first token.
- Report p50 and p99 end-to-end request latency.
- Report peak memory with `torch.cuda.max_memory_allocated()` on GPU or process RSS on CPU.
- Reset peak-memory counters consistently before each measured run.

## 3. Preserve Every Baseline

- Place the labeled vanilla Transformers `model.generate()` baseline in the first implementation column of every comparison table.
- Add each mini-vllm variant, including FP16, every quantized path, and every scheduler policy, as a new column.
- Never replace or remove an earlier column; show the full performance journey.
- Give every implementation the same tokenized workload, stopping rules, and output-token budget.

## 4. Record the Environment

- Record the GPU or CPU model, dtype, PyTorch version, KV block size, and whether the host is a free-tier instance subject to thermal throttling.
- Record the checkpoint identity, quantization mode, relevant CUDA versions, dependency versions, prompt and output lengths, scheduler limits, concurrency, warmup count, timed-run count, and synchronization policy.
- Treat every number without its hardware context as invalid.
- Record accuracy evidence and known parity drift alongside any quantized result.

## 5. Publish the Results

- Append a dated Markdown heading and comparison table to `docs/results.md`; never overwrite prior results.
- Add one sentence stating what changed since the previous row.
- Include the complete environment and workload conditions with the table, and keep the raw measurements in the harness-produced output.
- Flag every regression explicitly in the table instead of hiding, omitting, or selectively reframing it.
- Publish numbers only when the compared workloads and conditions are equivalent.

## 6. Enforce the Evidence Rule

- Delete any performance claim that did not come from this versioned benchmark harness.
- Reject performance numbers in a README, results table, commit message, or milestone summary until the complete protocol and metric set pass.
