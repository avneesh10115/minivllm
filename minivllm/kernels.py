"""Triton kernel that runs decode attention for the whole batch at once.

The plain PyTorch path calls attention once per sequence because every sequence
has its own block table, so a step with 32 sequences makes 32 small calls in
Python. This kernel follows the block table inside the GPU instead, so one
launch covers the batch. Importing this module needs Triton, so callers should
catch ImportError and fall back to the PyTorch path.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def paged_decode_kernel(
    query_ptr,
    cache_ptr,
    table_ptr,
    length_ptr,
    out_ptr,
    scale,
    query_seq_stride,
    query_head_stride,
    cache_kv_stride,
    cache_block_stride,
    cache_slot_stride,
    cache_head_stride,
    table_seq_stride,
    out_seq_stride,
    out_head_stride,
    head_dim,
    block_size,
    PADDED_HEAD_DIM: tl.constexpr,
    PADDED_BLOCK: tl.constexpr,
):
    seq = tl.program_id(0)
    head = tl.program_id(1)
    length = tl.load(length_ptr + seq)

    dims = tl.arange(0, PADDED_HEAD_DIM)
    dim_ok = dims < head_dim
    query = tl.load(
        query_ptr + seq * query_seq_stride + head * query_head_stride + dims,
        mask=dim_ok,
        other=0.0,
    ).to(tl.float32)

    slots = tl.arange(0, PADDED_BLOCK)
    running_max = float("-inf")
    running_sum = 0.0
    total = tl.zeros([PADDED_HEAD_DIM], dtype=tl.float32)

    for block_index in range(tl.cdiv(length, block_size)):
        block_id = tl.load(table_ptr + seq * table_seq_stride + block_index)
        positions = block_index * block_size + slots
        slot_ok = (slots < block_size) & (positions < length)

        base = cache_ptr + block_id * cache_block_stride + head * cache_head_stride
        offsets = slots[:, None] * cache_slot_stride + dims[None, :]
        load_ok = slot_ok[:, None] & dim_ok[None, :]
        keys = tl.load(base + offsets, mask=load_ok, other=0.0).to(tl.float32)
        values = tl.load(
            base + cache_kv_stride + offsets, mask=load_ok, other=0.0
        ).to(tl.float32)

        scores = tl.sum(keys * query[None, :], axis=1) * scale
        scores = tl.where(slot_ok, scores, float("-inf"))

        # Running softmax, so the kernel never holds every score at once.
        new_max = tl.maximum(running_max, tl.max(scores, axis=0))
        correction = tl.exp(running_max - new_max)
        weights = tl.exp(scores - new_max)
        total = total * correction + tl.sum(weights[:, None] * values, axis=0)
        running_sum = running_sum * correction + tl.sum(weights, axis=0)
        running_max = new_max

    result = total / running_sum
    tl.store(
        out_ptr + seq * out_seq_stride + head * out_head_stride + dims,
        result.to(out_ptr.dtype.element_ty),
        mask=dim_ok,
    )


def paged_decode_attention(
    query: torch.Tensor,
    cache: torch.Tensor,
    table: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """One decode step of attention for a batch of sequences.

    query is [batch, heads, head_dim], cache is the paged cache for one layer,
    table is the padded block table and lengths counts the cached tokens per
    sequence including the token just written.
    """
    batch, num_heads, head_dim = query.shape
    block_size = cache.shape[2]
    out = torch.empty(query.shape, dtype=query.dtype, device=query.device)
    paged_decode_kernel[(batch, num_heads)](
        query,
        cache,
        table,
        lengths,
        out,
        1.0 / (head_dim**0.5),
        query.stride(0),
        query.stride(1),
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        cache.stride(3),
        table.stride(0),
        out.stride(0),
        out.stride(1),
        head_dim,
        block_size,
        PADDED_HEAD_DIM=triton.next_power_of_2(head_dim),
        PADDED_BLOCK=triton.next_power_of_2(block_size),
    )
    return out
