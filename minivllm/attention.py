"""Reads and writes the paged KV cache and calculates attention."""

import math

import torch
import torch.nn.functional as F

try:
    from .kernels import paged_decode_attention

    HAS_TRITON = True
except ImportError:  # Triton is not installed on every platform.
    HAS_TRITON = False


def write_kv(
    cache: torch.Tensor,
    block_table: list[int],
    start_pos: int,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> None:
    block_size = cache.shape[2]
    num_new_tokens = keys.shape[0]
    positions = torch.arange(start_pos, start_pos + num_new_tokens, device=cache.device)
    table = torch.as_tensor(block_table, device=cache.device, dtype=torch.long)
    block_ids = table[positions // block_size]
    offsets = positions % block_size
    cache[0, block_ids, offsets] = keys.to(cache.dtype)
    cache[1, block_ids, offsets] = values.to(cache.dtype)


def read_kv(
    cache: torch.Tensor, block_table: list[int], length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    table = torch.as_tensor(block_table, device=cache.device, dtype=torch.long)
    gathered = cache[:, table].flatten(1, 2)
    return gathered[0, :length], gathered[1, :length]


def decode_attention(
    query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    scale = 1.0 / math.sqrt(query.shape[-1])
    scores = torch.einsum("hd,thd->ht", query, keys) * scale
    weights = torch.softmax(scores.float(), dim=-1).to(values.dtype)
    return torch.einsum("ht,thd->hd", weights, values)


def prefill_attention(
    query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    queries = query.transpose(0, 1)
    key_states = keys.transpose(0, 1)
    value_states = values.transpose(0, 1)
    attention = F.scaled_dot_product_attention(
        queries, key_states, value_states, is_causal=True
    )
    return attention.transpose(0, 1)


def build_decode_table(
    block_tables: list[list[int]], cache_lens: list[int], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Packs the batch's block tables into one padded tensor for the kernel."""
    width = max(len(table) for table in block_tables)
    padded = [table + [0] * (width - len(table)) for table in block_tables]
    table = torch.tensor(padded, dtype=torch.int32, device=device)
    # The kernel wants the length after this step's token is written.
    lengths = torch.tensor(
        [length + 1 for length in cache_lens], dtype=torch.int32, device=device
    )
    return table, lengths


def write_kv_decode(
    cache: torch.Tensor,
    table: torch.Tensor,
    positions: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
) -> None:
    """Writes one new token per sequence, one call for the whole batch."""
    block_size = cache.shape[2]
    rows = torch.arange(table.shape[0], device=cache.device)
    block_ids = table[rows, positions // block_size].long()
    offsets = positions % block_size
    cache[0, block_ids, offsets] = keys.to(cache.dtype)
    cache[1, block_ids, offsets] = values.to(cache.dtype)


def batched_decode_attention(
    query: torch.Tensor,
    cache: torch.Tensor,
    table: torch.Tensor,
    lengths: torch.Tensor,
    use_triton: bool = True,
) -> torch.Tensor:
    """Decode attention for every sequence in the batch."""
    if use_triton and HAS_TRITON and query.is_cuda:
        return paged_decode_attention(query, cache, table, lengths)
    result = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    for index in range(query.shape[0]):
        length = int(lengths[index])
        block_table = table[index].tolist()
        keys, values = read_kv(cache, block_table, length)
        result[index] = decode_attention(query[index], keys, values)
    return result
