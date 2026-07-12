"""Fixed-size paged KV storage and explicit sequence block tables."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Hashable, Sequence

import torch

from engine.config import EngineConfig

SequenceId = Hashable


class CacheExhaustedError(RuntimeError):
    """Raised when an append cannot reserve every required physical block."""


class BlockAllocator:
    """Deterministic refcounted allocator backed by an O(1) LIFO free list."""

    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self._refcounts = [0] * num_blocks
        # Reversing makes the first allocations 0, 1, ... while retaining LIFO reuse.
        self._free_blocks = list(reversed(range(num_blocks)))

    @property
    def num_blocks(self) -> int:
        return len(self._refcounts)

    @property
    def free_count(self) -> int:
        return len(self._free_blocks)

    def refcount(self, block_id: int) -> int:
        self._validate_block_id(block_id)
        return self._refcounts[block_id]

    def allocate(self) -> int:
        return self.allocate_many(1)[0]

    def allocate_many(self, count: int) -> list[int]:
        if count < 0:
            raise ValueError("allocation count must be non-negative")
        if count > self.free_count:
            raise CacheExhaustedError(
                f"requested {count} blocks with only {self.free_count} free"
            )

        allocated = [self._free_blocks.pop() for _ in range(count)]
        for block_id in allocated:
            self._refcounts[block_id] = 1
        return allocated

    def incref(self, block_id: int) -> None:
        self._validate_allocated(block_id)
        self._refcounts[block_id] += 1

    def free(self, block_id: int) -> None:
        self._validate_allocated(block_id)
        self._refcounts[block_id] -= 1
        if self._refcounts[block_id] == 0:
            self._free_blocks.append(block_id)

    def assert_no_leaks(self) -> None:
        leaked = [
            block_id
            for block_id, refcount in enumerate(self._refcounts)
            if refcount != 0
        ]
        if leaked:
            raise RuntimeError(f"leaked KV blocks: {leaked}")

    def _validate_block_id(self, block_id: int) -> None:
        if block_id < 0 or block_id >= self.num_blocks:
            raise IndexError(f"block_id {block_id} is out of range")

    def _validate_allocated(self, block_id: int) -> None:
        self._validate_block_id(block_id)
        if self._refcounts[block_id] == 0:
            raise RuntimeError(f"block {block_id} is not allocated")


@dataclass
class SequenceBlockTable:
    """Readable logical sequence state; tensor metadata is materialized on append."""

    block_ids: list[int] = field(default_factory=list)
    num_tokens: int = 0


@dataclass(frozen=True)
class LayerKVBlocks:
    """Physical K/V storage for one decoder layer."""

    key: torch.Tensor
    value: torch.Tensor


@dataclass
class AppendReservation:
    """An all-or-nothing append reserved across one model batch."""

    sequence_ids: tuple[SequenceId, ...]
    start_lengths: tuple[int, ...]
    end_lengths: tuple[int, ...]
    new_blocks: tuple[tuple[int, ...], ...]
    block_tables: torch.Tensor
    context_lengths: torch.Tensor
    query_start_positions: torch.Tensor
    owner_id: int
    state: str = "open"


class PagedKVCache:
    """Own physical layer storage and logical per-sequence block tables."""

    def __init__(self, config: EngineConfig) -> None:
        if config.num_kv_blocks <= 0:
            raise ValueError("num_kv_blocks must be positive to create a KV cache")
        self.config = config
        self.allocator = BlockAllocator(config.num_kv_blocks)
        self._sequences: dict[SequenceId, SequenceBlockTable] = {}
        self._active_sequences: set[SequenceId] = set()

        block_shape = (
            config.num_kv_blocks,
            config.kv_block_size,
            config.num_key_value_heads,
            config.head_dim,
        )
        device = torch.device(config.device)
        self.layers = tuple(
            LayerKVBlocks(
                key=torch.empty(block_shape, dtype=config.torch_dtype, device=device),
                value=torch.empty(block_shape, dtype=config.torch_dtype, device=device),
            )
            for _ in range(config.num_hidden_layers)
        )

    def create_sequence(self, sequence_id: SequenceId) -> None:
        if sequence_id in self._sequences:
            raise ValueError(f"sequence {sequence_id!r} already exists")
        self._sequences[sequence_id] = SequenceBlockTable()

    def sequence_state(self, sequence_id: SequenceId) -> SequenceBlockTable:
        try:
            return self._sequences[sequence_id]
        except KeyError as error:
            raise KeyError(f"unknown sequence {sequence_id!r}") from error

    def has_sequence(self, sequence_id: SequenceId) -> bool:
        """Return whether logical cache state exists for a sequence."""

        return sequence_id in self._sequences

    @property
    def free_block_count(self) -> int:
        return self.allocator.free_count

    def required_blocks_for_append(
        self,
        sequence_id: SequenceId,
        query_length: int,
    ) -> int:
        """Return the additional physical blocks needed for one append."""

        if query_length <= 0:
            raise ValueError("query_length must be positive")
        state = self.sequence_state(sequence_id)
        end_length = state.num_tokens + query_length
        if end_length > self.config.max_position_embeddings:
            raise ValueError("cache append exceeds the configured context length")
        return math.ceil(end_length / self.config.kv_block_size) - len(
            state.block_ids
        )

    def append_capacity(self, sequence_id: SequenceId) -> int:
        """Return tokens appendable using this sequence's tail and all free blocks."""

        state = self.sequence_state(sequence_id)
        allocated_slots = len(state.block_ids) * self.config.kv_block_size
        unused_tail_slots = allocated_slots - state.num_tokens
        free_block_slots = self.free_block_count * self.config.kv_block_size
        context_slots = self.config.max_position_embeddings - state.num_tokens
        return min(unused_tail_slots + free_block_slots, context_slots)

    def begin_append(
        self,
        sequence_ids: Sequence[SequenceId],
        query_length: int,
    ) -> AppendReservation:
        ids = tuple(sequence_ids)
        if not ids:
            raise ValueError("an append batch cannot be empty")
        if len(set(ids)) != len(ids):
            raise ValueError("sequence IDs in one append batch must be unique")
        if query_length <= 0:
            raise ValueError("query_length must be positive")
        if any(sequence_id in self._active_sequences for sequence_id in ids):
            raise RuntimeError("a sequence already has an active append")

        states = [self.sequence_state(sequence_id) for sequence_id in ids]
        start_lengths = tuple(state.num_tokens for state in states)
        end_lengths = tuple(start + query_length for start in start_lengths)
        if any(end > self.config.max_position_embeddings for end in end_lengths):
            raise ValueError("cache append exceeds the configured context length")

        required_counts = tuple(
            self.required_blocks_for_append(sequence_id, query_length)
            for sequence_id in ids
        )
        allocated = self.allocator.allocate_many(sum(required_counts))
        new_blocks: list[tuple[int, ...]] = []
        cursor = 0
        try:
            for state, count in zip(states, required_counts, strict=True):
                assigned = tuple(allocated[cursor : cursor + count])
                cursor += count
                state.block_ids.extend(assigned)
                new_blocks.append(assigned)

            max_blocks = max(len(state.block_ids) for state in states)
            device = torch.device(self.config.device)
            block_tables = torch.full(
                (len(states), max_blocks),
                -1,
                dtype=torch.long,
                device=device,
            )
            for row, state in enumerate(states):
                block_tables[row, : len(state.block_ids)] = torch.tensor(
                    state.block_ids, dtype=torch.long, device=device
                )

            reservation = AppendReservation(
                sequence_ids=ids,
                start_lengths=start_lengths,
                end_lengths=end_lengths,
                new_blocks=tuple(new_blocks),
                block_tables=block_tables,
                context_lengths=torch.tensor(end_lengths, dtype=torch.long, device=device),
                query_start_positions=torch.tensor(
                    start_lengths, dtype=torch.long, device=device
                ),
                owner_id=id(self),
            )
            self._active_sequences.update(ids)
            return reservation
        except Exception:
            for state, assigned in zip(states, new_blocks, strict=False):
                if assigned:
                    del state.block_ids[-len(assigned) :]
            for block_id in reversed(allocated):
                self.allocator.free(block_id)
            raise

    def write(
        self,
        layer_index: int,
        reservation: AppendReservation,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        self._validate_open(reservation)
        expected_shape = (
            len(reservation.sequence_ids),
            self.config.num_key_value_heads,
            reservation.end_lengths[0] - reservation.start_lengths[0],
            self.config.head_dim,
        )
        if key.shape != expected_shape or value.shape != expected_shape:
            raise ValueError(
                f"K/V append shape must be {expected_shape}, got "
                f"key={tuple(key.shape)}, value={tuple(value.shape)}"
            )
        if key.dtype != self.config.torch_dtype or value.dtype != self.config.torch_dtype:
            raise ValueError("K/V dtype must match EngineConfig")
        if layer_index < 0 or layer_index >= len(self.layers):
            raise IndexError(f"layer_index {layer_index} is out of range")
        layer = self.layers[layer_index]

        query_length = expected_shape[2]
        # Cache state is inference-only; detach writes from any accidental grad graph.
        with torch.no_grad():
            for row, sequence_id in enumerate(reservation.sequence_ids):
                state = self.sequence_state(sequence_id)
                for query_offset in range(query_length):
                    position = reservation.start_lengths[row] + query_offset
                    logical_block = position // self.config.kv_block_size
                    block_offset = position % self.config.kv_block_size
                    physical_block = state.block_ids[logical_block]
                    layer.key[physical_block, block_offset].copy_(
                        key[row, :, query_offset, :]
                    )
                    layer.value[physical_block, block_offset].copy_(
                        value[row, :, query_offset, :]
                    )

    def commit(self, reservation: AppendReservation) -> None:
        self._validate_open(reservation)
        for sequence_id, end_length in zip(
            reservation.sequence_ids, reservation.end_lengths, strict=True
        ):
            self.sequence_state(sequence_id).num_tokens = end_length
        reservation.state = "committed"
        self._active_sequences.difference_update(reservation.sequence_ids)

    def rollback(self, reservation: AppendReservation) -> None:
        self._validate_open(reservation)
        for sequence_id, assigned in zip(
            reservation.sequence_ids, reservation.new_blocks, strict=True
        ):
            state = self.sequence_state(sequence_id)
            if assigned:
                del state.block_ids[-len(assigned) :]
        for assigned in reversed(reservation.new_blocks):
            for block_id in reversed(assigned):
                self.allocator.free(block_id)
        reservation.state = "rolled_back"
        self._active_sequences.difference_update(reservation.sequence_ids)

    def release_sequence(self, sequence_id: SequenceId) -> None:
        if sequence_id in self._active_sequences:
            raise RuntimeError("cannot release a sequence with an active append")
        state = self.sequence_state(sequence_id)
        for block_id in reversed(state.block_ids):
            self.allocator.free(block_id)
        del self._sequences[sequence_id]

    def assert_no_leaks(self) -> None:
        if self._sequences:
            raise RuntimeError(f"live cache sequences: {list(self._sequences)}")
        self.allocator.assert_no_leaks()

    def _validate_open(self, reservation: AppendReservation) -> None:
        if reservation.state != "open":
            raise RuntimeError(f"append reservation is {reservation.state}")
        if reservation.owner_id != id(self):
            raise RuntimeError("append reservation belongs to a different cache")
        if not set(reservation.sequence_ids).issubset(self._active_sequences):
            raise RuntimeError("append reservation does not belong to this cache")


def paged_attention(
    query: torch.Tensor,
    *,
    key_blocks: torch.Tensor,
    value_blocks: torch.Tensor,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    query_start_positions: torch.Tensor,
    block_size: int,
    query_heads_per_kv_head: int,
    scaling: float,
) -> torch.Tensor:
    """Pure-PyTorch backend boundary from paged K/V state to attention output."""

    batch, _, query_length, _ = query.shape
    if block_tables.shape[0] != batch:
        raise ValueError("block_tables batch dimension must match query")
    if context_lengths.shape != (batch,) or query_start_positions.shape != (batch,):
        raise ValueError("context lengths and query starts must have shape [batch]")

    outputs: list[torch.Tensor] = []
    for row in range(batch):
        context_length = int(context_lengths[row])
        query_start = int(query_start_positions[row])
        physical_count = math.ceil(context_length / block_size)
        physical_ids = block_tables[row, :physical_count]

        keys = key_blocks.index_select(0, physical_ids)
        values = value_blocks.index_select(0, physical_ids)
        keys = keys.reshape(-1, keys.shape[-2], keys.shape[-1])[:context_length]
        values = values.reshape(-1, values.shape[-2], values.shape[-1])[:context_length]
        keys = keys.permute(1, 0, 2).unsqueeze(0)
        values = values.permute(1, 0, 2).unsqueeze(0)
        keys = _repeat_paged_kv(keys, query_heads_per_kv_head)
        values = _repeat_paged_kv(values, query_heads_per_kv_head)

        row_query = query[row : row + 1]
        scores = torch.matmul(row_query, keys.transpose(-2, -1)) * scaling
        query_positions = torch.arange(
            query_start,
            query_start + query_length,
            device=query.device,
        )
        key_positions = torch.arange(context_length, device=query.device)
        blocked = key_positions[None, :] > query_positions[:, None]
        scores = scores.masked_fill(blocked[None, None], torch.finfo(scores.dtype).min)
        # Match the dense reference's stable FP32 softmax policy.
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
        outputs.append(torch.matmul(probabilities, values))

    return torch.cat(outputs, dim=0)


def _repeat_paged_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden_states
    batch, kv_heads, sequence, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None, :, :].expand(
        batch, kv_heads, repeats, sequence, head_dim
    )
    return expanded.reshape(batch, kv_heads * repeats, sequence, head_dim)
