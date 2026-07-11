---
name: hf-parity-testing
description: How to write and run correctness tests that verify mini-vllm matches the HuggingFace reference token-for-token. Use this skill whenever writing or modifying ANY model component (attention, RMSNorm, RoPE, SwiGLU, KV cache, sampling), whenever a test fails with numerical mismatch, and before merging any change that touches the forward pass or decode loop, even if the user doesn't say "test".
---

# Hugging Face Parity Testing

## Build the two-tier parity suite

1. Add unit-level tensor parity for every changed module.
   - Instantiate the mini-vllm module and its Hugging Face counterpart.
   - Load identical weights and feed identical inputs, masks, positions, and cache state.
   - Compare outputs with `torch.testing.assert_close`.
   - State `rtol` and `atol` explicitly at every assertion; use these defaults unless a tighter module-specific tolerance passes reliably:
     - CPU FP32: `rtol=1e-5`, `atol=1e-6`.
     - GPU FP16: `rtol=1e-3`, `atol=1e-3`.
     - GPU BF16: `rtol=1e-2`, `atol=1e-2`.
   - Cover attention, RMSNorm, RoPE, SwiGLU, KV-cache updates and reads, and sampler logits or greedy selection whenever they change.
2. Add end-to-end greedy-decode parity.
   - Keep exactly five fixed prompt fixtures covering short, long, punctuation-heavy, repeated-token, and boundary-sensitive input.
   - Use the same tokenizer, input token IDs, stopping conditions, and output-token limit for both implementations.
   - Set `temperature=0` and disable every sampling path.
   - Compare generated token IDs at every decode step and require exact sequence equality; do not compare decoded text as a substitute.

## Enforce determinism

1. Set fixed seeds for Python, PyTorch, and every active CUDA device before constructing modules, inputs, or caches.
2. Treat CPU FP32 results as the correctness ground truth.
3. Call `torch.use_deterministic_algorithms(True)` where practical and document any operation that cannot use a deterministic implementation.
4. Reset model state, KV-cache state, and random state between reference and mini-vllm runs.
5. Never compare sampled outputs. Test sampling components with fixed tensors and controlled random inputs, but use greedy decoding for token-for-token parity.

## Check Qwen3 divergence traps first

1. Verify grouped-query attention uses 16 query heads and 8 key/value heads.
2. Verify each key/value head is repeated for the correct query-head group before the attention dot product, without changing head order.
3. Verify QK-Norm is applied independently per head to Q and K after projection and before RoPE.
4. Verify RoPE uses the checkpoint's theta, position indexing, rotary dimensions, and Hugging Face interleaving convention.
5. Verify the output projection receives heads in the same order and layout as Hugging Face.
6. Verify input embeddings and the language-model head follow the checkpoint's weight-tying configuration and share storage when tying is enabled.
7. Verify every safetensors weight maps to the intended parameter shape; check projection orientation and apply a transpose only when the mini-vllm layout differs from the stored Hugging Face layout.

## Bisect numerical mismatches

1. Run the Hugging Face model with `output_hidden_states=True` on the failing fixed input.
2. Capture the matching mini-vllm embedding output and every decoder-layer output.
3. Compare hidden states in order and identify the first diverging layer; ignore later divergence until that layer matches.
4. Capture and compare checkpoints inside the first diverging layer in this order:
   - Pre-attention normalization.
   - Q, K, and V projections after head reshaping.
   - Q and K after QK-Norm and after RoPE.
   - Attention probabilities or reference attention result, then attention output projection and residual.
   - Post-attention normalization, SwiGLU gate/up activations, down projection, and final residual.
5. Reduce the failing input to the shortest sequence and smallest cache state that preserves the first mismatch.
6. Fix the earliest mismatching checkpoint, then rerun the unit parity test, the five-prompt greedy suite, and the full parity suite.

## Enforce the parity gate

1. Run focused parity tests after each model or decoding change and run `python -m pytest -q tests/parity` before merging.
2. Do not advance a milestone or merge a pull request while any parity test is red.
3. Do not loosen a tolerance to hide a mismatch.
4. Loosen a tolerance only after adding a `docs/decisions.md` entry that explains the numerical cause, affected dtype and hardware, old and new tolerances, measured evidence, and why the new bound still catches incorrect behavior.
