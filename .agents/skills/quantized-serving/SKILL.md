---
name: quantized-serving
description: How to quantize Qwen3-0.6B and serve it through mini-vllm with a rigorous accuracy/latency report. Use this skill whenever quantization, INT8, FP8, INT4, compression, calibration, or "compressed model" comes up, whenever Milestone 6 work starts, and whenever anyone proposes a speed number for the quantized path, even if accuracy isn't mentioned.
---

# Quantized Serving

Use `Qwen/Qwen3-0.6B-Base` as the canonical checkpoint. Treat prior Sconce
compression results as motivation only; never transfer its memory or accuracy
numbers to mini-vllm without measurements from this repository.

## 1. Follow the quantization ladder

1. Implement INT8 weight-only quantization with symmetric per-output-channel
   scales first. Derive every scale from the weights; require no calibration
   dataset for this rung.
2. Establish the weight-only accuracy and performance baseline before optionally
   adding INT8 dynamic activation quantization.
3. Document KV-cache quantization as the next rung after v1. Do not include it in
   the v1 quantized-serving goal.
4. Treat FP8, INT4, and other compressed formats as later rungs. Require each
   rung to pass this same accuracy and reporting protocol.

## 2. Implement the reference path explicitly

1. Compute one symmetric scale for each output channel from that channel's
   absolute maximum. Quantize into the signed INT8 range and handle an all-zero
   channel without dividing by zero.
2. Store the quantized weights and their per-channel scales explicitly. Keep
   tensor shapes and the scale broadcasting dimension visible in code.
3. Dequantize the weight into the activation compute dtype immediately before
   `F.linear` or the equivalent matmul in the pure-PyTorch reference path, then
   perform the matmul in floating point.
4. Preserve this readable dequantize-then-matmul path as the correctness oracle
   for any later fused kernel. Do not imply that weight storage savings guarantee
   a latency improvement.
5. Keep RMSNorm, token embeddings, and the tied LM head in FP16 on the T4 and in
   FP32 for CPU correctness first. Document that RMSNorm offers little weight
   memory benefit and is numerically sensitive, while embeddings and the LM head
   directly affect token probabilities and share tied weights.
6. Route the quantized model through the existing cache, scheduler, sampler, and
   server interfaces. Do not fork serving behavior by quantization mode.

## 3. Inspect outliers before broadening quantization

1. Measure activation ranges by channel on the fixed evaluation workload before
   trusting any per-tensor weight or activation scheme.
2. Record per-channel minima, maxima, absolute maxima, and a robust percentile so
   outlier channels remain visible rather than being hidden by one aggregate.
3. Use the observed outliers to explain why the first weight path uses
   per-channel scales. Record the measured ranges in `docs/results.md` and their
   design implications in `docs/decisions.md`.
4. Reject a per-tensor proposal that lacks an outlier analysis and an explicit
   accuracy comparison with the per-channel reference.

## 4. Run the non-negotiable accuracy protocol

1. Freeze one versioned evaluation text set before comparing FP16 and quantized
   models. Use that exact set forever; do not edit or replace it between rows.
2. Measure perplexity for the FP16 engine and every quantized configuration on
   that set. Report both absolute perplexity and the delta from FP16.
3. Freeze 20 held-out prompts as a small task-quality check. Run identical
   decoding settings, compare answers qualitatively, and record prompt-level
   outcomes plus an overall outcome without hiding regressions.
4. Greedily decode the fixed drift prompts with the FP16 and quantized engines.
   Report the fraction of generated token IDs that match at the same position;
   count divergent or missing positions and early or late EOS as mismatches.
5. Report per-prompt and aggregate greedy-decode drift. Do not require exact
   parity; record the accepted tolerance and its rationale in
   `docs/decisions.md` before accepting the quantized path.

## 5. Report one Pareto row per result

1. Read and follow the `inference-benchmark` skill before measuring or publishing
   quantized performance.
2. Add one Pareto row for every quantized configuration. Put tokens/second, peak
   memory, TTFT, p50 latency, and p99 latency from that skill in the same table as
   absolute perplexity, perplexity delta, the 20-prompt task outcome, and greedy
   token-match drift.
3. Keep workload, checkpoint, dtype, hardware, concurrency, scheduler limits,
   warmup, synchronization, and run-count conditions equivalent across rows.
4. Delete any quantized speed number whose row lacks its accuracy columns. Never
   publish speed and accuracy in separate comparisons.
5. Describe a result as better only when the table shows the latency, throughput,
   memory, and accuracy tradeoff. Do not call an unmeasured result Pareto-optimal.

## 6. Gate completion

1. Compare quantized and FP16 logits on fixed inputs, run the frozen accuracy
   protocol, and test end-to-end serving through the shared interfaces.
2. Run the focused tests, the Hugging Face parity suite, and the versioned
   benchmark harness required by `AGENTS.md`.
3. Update `docs/decisions.md` and `docs/results.md` with measured evidence. Do not
   declare Milestone 6 complete while a required test fails or a Pareto row is
   missing accuracy evidence.
