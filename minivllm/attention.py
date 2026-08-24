"""Reads and writes the paged KV cache and calculates attention."""

import math

import torch
import torch.nn.functional as F


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
