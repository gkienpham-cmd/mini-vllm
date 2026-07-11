"""Generation loops implemented without Transformers generation utilities."""

from __future__ import annotations

import torch

from engine.model.qwen3 import Qwen3ForCausalLM


@torch.inference_mode()
def greedy_decode(
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    *,
    max_new_tokens: int,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Append the highest-logit token at every step.

    Milestone 1 intentionally recomputes the whole prefix. This slow path is the
    correctness oracle for the paged KV cache introduced in Milestone 2.
    """

    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape [batch, sequence]")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    if input_ids.shape[1] + max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("requested generation exceeds the configured context length")

    token_ids = input_ids
    finished = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)

    for _ in range(max_new_tokens):
        logits = model(token_ids).logits
        next_token = logits[:, -1, :].argmax(dim=-1)

        if eos_token_id is not None:
            # Finished rows keep emitting EOS so batched tensors stay rectangular.
            eos_tokens = torch.full_like(next_token, eos_token_id)
            next_token = torch.where(finished, eos_tokens, next_token)
            finished = finished | next_token.eq(eos_token_id)

        token_ids = torch.cat((token_ids, next_token[:, None]), dim=1)
        if eos_token_id is not None and bool(finished.all()):
            break

    return token_ids

