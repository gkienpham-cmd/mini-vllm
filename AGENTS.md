# mini-vllm Contributor Contract

This repository is a teaching-first LLM inference engine. Correctness comes
before optimization, and measured evidence comes before performance claims.
The canonical target is `Qwen/Qwen3-0.6B-Base`; do not silently substitute the
instruction-tuned checkpoint.

## How We Work

- Prefer readable code over clever code. Keep tensor shapes explicit, choose
  descriptive names, and add a short comment explaining *why* any non-obvious
  line exists.
- Build one milestone at a time. Before implementation, explain the approach
  and data structures, present two or three meaningful options at each real
  design fork, and obtain the user's choice.
- After implementation, run the required tests, explain the design and measured
  result, and quiz the user on the core systems idea. Do not begin the next
  milestone until the previous milestone's tests are green and the user can
  defend the design.
- Find the limiter before optimizing it. Preserve simple reference paths so an
  optimization can always be checked against a known-correct implementation.

## Repository Shape and Configuration

The planned top-level layout is:

```text
engine/          Model, attention, KV cache, scheduling, sampling, quantization
server/          FastAPI transport and OpenAI-compatible schemas
bench/           Versioned performance and accuracy harnesses
tests/           Unit, parity, integration, and fixture code
docs/            Decision records and measured results
.agents/skills/  Project-specific contributor workflows
```

These are root-level packages; do not introduce an additional
`src/mini_vllm/` package layer.

`engine/config.py` will define the sole shared, immutable `EngineConfig` object.
It will combine architecture values derived from the checkpoint with explicit
runtime choices such as device, dtype, KV block size, token budget, and
attention backend. Model, cache, scheduler, server, benchmark, and test code
must consume this object rather than maintain shadow configuration dictionaries
or duplicate constants.

## Quality Gates

Run checks from the repository root:

```bash
# Full portable suite.
python -m pytest -q

# CPU correctness without CUDA or benchmark-only tests.
python -m pytest -q -m "not cuda and not benchmark"

# Hugging Face parity suite.
python -m pytest -q tests/parity
```

- Do not merge a change or declare a milestone complete while a required test
  is failing.
- Any change that can affect model outputs requires a green Hugging Face parity
  suite, in addition to its focused tests.
- CPU FP32 correctness is the portable baseline and must remain runnable on
  machines without a GPU.
- Benchmarks run only through versioned scripts under `bench/`. Ad hoc timings
  are useful for debugging but are not publishable evidence.

## Benchmark Discipline

Do not make a performance claim without recording all of the following through
the benchmark harness:

- Hardware and software environment, including GPU model, dependency versions,
  and relevant CUDA versions.
- Model/checkpoint identity, dtype, quantization mode, workload, input/output
  lengths, concurrency, and scheduler limits.
- Warmup policy, synchronization policy, number of measured runs, and raw
  measurements—not only aggregates.
- Tokens/second, time to first token, p50/p99 latency, and peak memory whenever
  the milestone supports those metrics.
- Accuracy evidence for quantized paths, including perplexity delta, the fixed
  task-quality check, and known parity drift.

The canonical accelerator is a Colab NVIDIA T4. Benchmark comparisons must use
equivalent workloads and report conditions that could otherwise make an
apples-to-oranges result look faster.

## Allowed and Banned Dependencies

The engine and serving path may use PyTorch, the Hugging Face tokenizer, and
safetensors loading.

The following are banned from the core engine and serving implementation:

- `model.generate()`.
- Transformers generation utilities.
- vLLM.
- Text Generation Inference (TGI).

Transformers may be used only as the correctness oracle in parity tests and as
the explicitly labeled baseline in benchmarks. The Hugging Face tokenizer is
limited to encoding, decoding, and applicable template handling; token
generation, scheduling, caching, sampling, and serving logic belong to this
repository.

## Dtype Policy

- CPU correctness tests use FP32.
- Colab T4 execution uses FP16 model weights, activations, and KV cache because
  the T4 is the canonical target and should not inherit the checkpoint's BF16
  declaration blindly.
- Numerically sensitive reductions may explicitly accumulate in FP32. Every
  promotion must include a short comment explaining the stability reason and a
  parity test covering the behavior.
- Do not silently select BF16 merely because the checkpoint metadata declares
  it.
- Every quantized dtype or kernel path requires an explicit accuracy comparison
  with the non-quantized reference.

## Documentation and Skills

Every completed milestone updates both:

- `docs/decisions.md`, using the `decisions-log` project skill.
- `docs/results.md`, using the `inference-benchmark` project skill.

Each decision entry records the options considered, the choice and reasoning,
measured evidence, what changes at 10x model size or 100x traffic, and a one- or
two-sentence interview soundbite.

Before working in one of these domains, read and follow the corresponding skill:

- `.agents/skills/hf-parity-testing/SKILL.md` for model or decoding parity.
- `.agents/skills/inference-benchmark/SKILL.md` for results and performance work.
- `.agents/skills/decisions-log/SKILL.md` for milestone decision records.
- `.agents/skills/quantized-serving/SKILL.md` for quantization and accuracy work.

The user will provide these skills. Do not create substitutes. If a required
skill is missing when its domain begins, stop before making domain changes and
ask the user to provide it.

## Attention Backend Boundary

Keep model semantics, KV-cache ownership, and scheduling independent from the
attention implementation. The pure-PyTorch paged-attention path is the
correctness reference. A future `apex-attention` CUDA backend must be able to
consume paged K/V state and return attention output without requiring scheduler
changes. Backend additions must pass the same parity tests.

