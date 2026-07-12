"""Deterministic, single-owner continuous batching scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import torch

from engine.cache import PagedKVCache, SequenceId
from engine.model.qwen3 import Qwen3ForCausalLM


class RequestStatus(str, Enum):
    WAITING = "waiting"
    PREFILLING = "prefilling"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED = "finished"
    FAILED = "failed"


class FinishReason(str, Enum):
    EOS = "eos"
    LENGTH = "length"
    ERROR = "error"


@dataclass
class SchedulerRequest:
    """Scheduler-owned request state retained across preemption."""

    request_id: SequenceId
    prompt_token_ids: tuple[int, ...]
    max_new_tokens: int
    eos_token_id: int | None
    arrival_index: int
    status: RequestStatus = RequestStatus.WAITING
    generated_token_ids: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0
    finish_reason: FinishReason | None = None
    error: str | None = None

    @property
    def all_token_ids(self) -> tuple[int, ...]:
        return self.prompt_token_ids + tuple(self.generated_token_ids)

    @property
    def is_terminal(self) -> bool:
        return self.status in {RequestStatus.FINISHED, RequestStatus.FAILED}


@dataclass(frozen=True)
class RequestOutput:
    request_id: SequenceId
    token_id: int | None
    status: RequestStatus
    finish_reason: FinishReason | None = None
    error: str | None = None


@dataclass(frozen=True)
class SchedulerStep:
    scheduled_token_count: int
    outputs: tuple[RequestOutput, ...]
    admitted_request_ids: tuple[SequenceId, ...]
    preempted_request_ids: tuple[SequenceId, ...]


class SchedulerExecutionError(RuntimeError):
    """Model failure with the successfully completed portion of the step."""

    def __init__(self, message: str, step: SchedulerStep) -> None:
        super().__init__(message)
        self.step = step


class _ModelExecutionFailure(Exception):
    def __init__(
        self,
        original_error: Exception,
        outputs: tuple[RequestOutput, ...],
    ) -> None:
        super().__init__(str(original_error))
        self.original_error = original_error
        self.outputs = outputs


class ContinuousBatchScheduler:
    """Run greedy paged inference with FIFO admission and recompute preemption."""

    def __init__(self, model: Qwen3ForCausalLM, cache: PagedKVCache) -> None:
        if model.config != cache.config:
            raise ValueError("model and cache must use the same EngineConfig")
        if model.config.num_kv_blocks <= 0:
            raise ValueError("scheduler requires a positive KV block count")
        if model.config.max_num_batched_tokens <= 0:
            raise ValueError("scheduler requires a positive token budget")

        self.model = model
        self.cache = cache
        self.config = model.config
        self._requests: dict[SequenceId, SchedulerRequest] = {}
        self._next_arrival_index = 0

    @property
    def has_unfinished_requests(self) -> bool:
        return any(not request.is_terminal for request in self._requests.values())

    @property
    def request_ids(self) -> tuple[SequenceId, ...]:
        requests = sorted(
            self._requests.values(), key=lambda request: request.arrival_index
        )
        return tuple(request.request_id for request in requests)

    def get_request(self, request_id: SequenceId) -> SchedulerRequest:
        try:
            return self._requests[request_id]
        except KeyError as error:
            raise KeyError(f"unknown request {request_id!r}") from error

    def submit(
        self,
        request_id: SequenceId,
        prompt_token_ids: Sequence[int] | torch.Tensor,
        *,
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> SchedulerRequest:
        if request_id in self._requests:
            raise ValueError(f"request {request_id!r} already exists")
        prompt = self._normalize_prompt(prompt_token_ids)
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if eos_token_id is not None and not 0 <= eos_token_id < self.config.vocab_size:
            raise ValueError("eos_token_id is outside the configured vocabulary")

        maximum_tokens = len(prompt) + max_new_tokens
        if maximum_tokens > self.config.max_position_embeddings:
            raise ValueError("request exceeds the configured context length")
        cache_token_capacity = self.config.num_kv_blocks * self.config.kv_block_size
        if maximum_tokens > cache_token_capacity:
            raise ValueError("one request cannot fit in the physical KV cache")

        request = SchedulerRequest(
            request_id=request_id,
            prompt_token_ids=prompt,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            arrival_index=self._next_arrival_index,
        )
        self._next_arrival_index += 1
        if max_new_tokens == 0:
            request.status = RequestStatus.FINISHED
            request.finish_reason = FinishReason.LENGTH
        self._requests[request_id] = request
        return request

    @torch.inference_mode()
    def step(self) -> SchedulerStep:
        budget = self.config.max_num_batched_tokens
        scheduled_tokens = 0
        outputs: list[RequestOutput] = []
        admitted: list[SequenceId] = []
        preempted: list[SequenceId] = []

        decode_batch = self._ordered_requests(RequestStatus.RUNNING)
        self._make_decode_batch_fit(decode_batch, preempted)
        if decode_batch:
            try:
                decode_outputs = self._run_decode_batch(decode_batch)
            except _ModelExecutionFailure as failure:
                outputs.extend(failure.outputs)
                self._raise_execution_error(
                    failure,
                    scheduled_tokens=scheduled_tokens,
                    outputs=outputs,
                    admitted=admitted,
                    preempted=preempted,
                )
            scheduled_tokens += len(decode_batch)
            outputs.extend(decode_outputs)

        while scheduled_tokens < budget:
            request = self._next_prefill_request()
            if request is None:
                break

            if not self.cache.has_sequence(request.request_id):
                if not self._make_room_for(request, preempted):
                    break
                self.cache.create_sequence(request.request_id)
                request.status = RequestStatus.PREFILLING
                admitted.append(request.request_id)

            remaining_prefix = len(request.all_token_ids) - request.num_computed_tokens
            if remaining_prefix <= 0:
                raise RuntimeError("prefill request has no uncomputed tokens")

            remaining_budget = budget - scheduled_tokens
            chunk_length = min(remaining_prefix, remaining_budget)
            while self.cache.append_capacity(request.request_id) == 0:
                if not self._make_room_for(request, preempted):
                    break
            capacity = self.cache.append_capacity(request.request_id)
            chunk_length = min(chunk_length, capacity)
            if chunk_length == 0:
                break

            try:
                output = self._run_prefill_chunk(request, chunk_length)
            except _ModelExecutionFailure as failure:
                outputs.extend(failure.outputs)
                self._raise_execution_error(
                    failure,
                    scheduled_tokens=scheduled_tokens,
                    outputs=outputs,
                    admitted=admitted,
                    preempted=preempted,
                )
            scheduled_tokens += chunk_length
            if output is not None:
                outputs.append(output)

        if scheduled_tokens > budget:
            raise RuntimeError("scheduler exceeded its token budget")
        return SchedulerStep(
            scheduled_token_count=scheduled_tokens,
            outputs=tuple(outputs),
            admitted_request_ids=tuple(admitted),
            preempted_request_ids=tuple(preempted),
        )

    def _raise_execution_error(
        self,
        failure: _ModelExecutionFailure,
        *,
        scheduled_tokens: int,
        outputs: list[RequestOutput],
        admitted: list[SequenceId],
        preempted: list[SequenceId],
    ) -> None:
        partial_step = SchedulerStep(
            scheduled_token_count=scheduled_tokens,
            outputs=tuple(outputs),
            admitted_request_ids=tuple(admitted),
            preempted_request_ids=tuple(preempted),
        )
        raise SchedulerExecutionError(str(failure), partial_step) from (
            failure.original_error
        )

    def _normalize_prompt(
        self, prompt_token_ids: Sequence[int] | torch.Tensor
    ) -> tuple[int, ...]:
        if isinstance(prompt_token_ids, torch.Tensor):
            if prompt_token_ids.ndim != 1:
                raise ValueError("prompt_token_ids tensor must be one-dimensional")
            raw_prompt = prompt_token_ids.tolist()
        else:
            raw_prompt = list(prompt_token_ids)
        if not raw_prompt:
            raise ValueError("prompt_token_ids cannot be empty")
        if any(
            isinstance(token_id, bool) or not isinstance(token_id, int)
            for token_id in raw_prompt
        ):
            raise ValueError("prompt_token_ids must contain integers")
        if any(not 0 <= token_id < self.config.vocab_size for token_id in raw_prompt):
            raise ValueError("prompt token is outside the configured vocabulary")
        return tuple(raw_prompt)

    def _ordered_requests(self, *statuses: RequestStatus) -> list[SchedulerRequest]:
        allowed = set(statuses)
        return sorted(
            (
                request
                for request in self._requests.values()
                if request.status in allowed
            ),
            key=lambda request: request.arrival_index,
        )

    def _next_prefill_request(self) -> SchedulerRequest | None:
        requests = self._ordered_requests(
            RequestStatus.WAITING,
            RequestStatus.PREFILLING,
            RequestStatus.PREEMPTED,
        )
        return requests[0] if requests else None

    def _resident_requests(self) -> list[SchedulerRequest]:
        return [
            request
            for request in self._requests.values()
            if self.cache.has_sequence(request.request_id)
        ]

    def _make_decode_batch_fit(
        self,
        decode_batch: list[SchedulerRequest],
        preempted: list[SequenceId],
    ) -> None:
        while decode_batch:
            required_blocks = sum(
                self.cache.required_blocks_for_append(request.request_id, 1)
                for request in decode_batch
            )
            if required_blocks <= self.cache.free_block_count:
                return
            victim = max(
                self._resident_requests(), key=lambda request: request.arrival_index
            )
            self._preempt(victim, preempted)
            if victim in decode_batch:
                decode_batch.remove(victim)

    def _make_room_for(
        self,
        request: SchedulerRequest,
        preempted: list[SequenceId],
    ) -> bool:
        """Let older work reclaim blocks without allowing it to be overtaken."""

        if self.cache.free_block_count > 0:
            return True
        younger_residents = [
            resident
            for resident in self._resident_requests()
            if resident.request_id != request.request_id
            and resident.arrival_index > request.arrival_index
        ]
        if not younger_residents:
            return self.cache.has_sequence(request.request_id) and (
                self.cache.append_capacity(request.request_id) > 0
            )
        victim = max(younger_residents, key=lambda item: item.arrival_index)
        self._preempt(victim, preempted)
        return True

    def _preempt(
        self,
        request: SchedulerRequest,
        preempted: list[SequenceId],
    ) -> None:
        if not self.cache.has_sequence(request.request_id):
            raise RuntimeError("cannot preempt a non-resident request")
        self.cache.release_sequence(request.request_id)
        request.num_computed_tokens = 0
        request.status = RequestStatus.PREEMPTED
        preempted.append(request.request_id)

    def _run_decode_batch(
        self,
        requests: list[SchedulerRequest],
    ) -> list[RequestOutput]:
        input_ids = torch.tensor(
            [[request.all_token_ids[request.num_computed_tokens]] for request in requests],
            dtype=torch.long,
            device=torch.device(self.config.device),
        )
        try:
            model_output = self.model.forward_cached(
                input_ids,
                cache=self.cache,
                sequence_ids=[request.request_id for request in requests],
            )
        except Exception as error:
            failed_outputs = []
            for request in requests:
                failed_outputs.append(self._fail_request(request, error))
            raise _ModelExecutionFailure(error, tuple(failed_outputs)) from error

        outputs: list[RequestOutput] = []
        for row, request in enumerate(requests):
            request.num_computed_tokens += 1
            token_id = int(model_output.logits[row, -1].argmax())
            outputs.append(self._record_generated_token(request, token_id))
        return outputs

    def _run_prefill_chunk(
        self,
        request: SchedulerRequest,
        chunk_length: int,
    ) -> RequestOutput | None:
        start = request.num_computed_tokens
        end = start + chunk_length
        input_ids = torch.tensor(
            [request.all_token_ids[start:end]],
            dtype=torch.long,
            device=torch.device(self.config.device),
        )
        try:
            model_output = self.model.forward_cached(
                input_ids,
                cache=self.cache,
                sequence_ids=[request.request_id],
            )
        except Exception as error:
            failed_output = self._fail_request(request, error)
            raise _ModelExecutionFailure(error, (failed_output,)) from error

        request.num_computed_tokens = end
        if request.num_computed_tokens < len(request.all_token_ids):
            request.status = RequestStatus.PREFILLING
            return None

        token_id = int(model_output.logits[0, -1].argmax())
        return self._record_generated_token(request, token_id)

    def _record_generated_token(
        self,
        request: SchedulerRequest,
        token_id: int,
    ) -> RequestOutput:
        request.generated_token_ids.append(token_id)
        finish_reason: FinishReason | None = None
        if request.eos_token_id is not None and token_id == request.eos_token_id:
            finish_reason = FinishReason.EOS
        elif len(request.generated_token_ids) >= request.max_new_tokens:
            finish_reason = FinishReason.LENGTH

        if finish_reason is not None:
            request.status = RequestStatus.FINISHED
            request.finish_reason = finish_reason
            self.cache.release_sequence(request.request_id)
        else:
            request.status = RequestStatus.RUNNING
        return RequestOutput(
            request_id=request.request_id,
            token_id=token_id,
            status=request.status,
            finish_reason=finish_reason,
        )

    def _fail_request(
        self, request: SchedulerRequest, error: Exception
    ) -> RequestOutput:
        if self.cache.has_sequence(request.request_id):
            self.cache.release_sequence(request.request_id)
        request.status = RequestStatus.FAILED
        request.finish_reason = FinishReason.ERROR
        request.error = str(error)
        return RequestOutput(
            request_id=request.request_id,
            token_id=None,
            status=RequestStatus.FAILED,
            finish_reason=FinishReason.ERROR,
            error=str(error),
        )
