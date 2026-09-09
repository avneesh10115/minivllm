import unittest

import torch

from minivllm.attention import (
    HAS_TRITON,
    batched_decode_attention,
    build_decode_table,
    write_kv_decode,
)

needs_gpu = unittest.skipUnless(
    HAS_TRITON and torch.cuda.is_available(), "needs a GPU with Triton"
)


def build_case(device, block_tables, cache_lens, num_heads=4, head_dim=64, block_size=16):
    """Fills a paged cache with random values and returns one decode step."""
    torch.manual_seed(0)
    num_blocks = max(max(table) for table in block_tables) + 1
    cache = torch.randn(
        2, num_blocks, block_size, num_heads, head_dim, device=device, dtype=torch.float16
    )
    batch = len(block_tables)
    query = torch.randn(batch, num_heads, head_dim, device=device, dtype=torch.float16)
    keys = torch.randn(batch, num_heads, head_dim, device=device, dtype=torch.float16)
    values = torch.randn(batch, num_heads, head_dim, device=device, dtype=torch.float16)
    table, lengths = build_decode_table(block_tables, cache_lens, device)
    positions = torch.tensor(cache_lens, device=device)
    write_kv_decode(cache, table, positions, keys, values)
    return query, cache, table, lengths


class PagedKernelTest(unittest.TestCase):
    @needs_gpu
    def test_kernel_matches_pytorch_on_ragged_batch(self):
        device = torch.device("cuda")
        # Different lengths, out of order blocks, and one shared block.
        block_tables = [[3, 1], [0], [2, 4, 1], [0, 3]]
        cache_lens = [20, 5, 33, 16]
        query, cache, table, lengths = build_case(device, block_tables, cache_lens)

        expected = batched_decode_attention(query, cache, table, lengths, use_triton=False)
        actual = batched_decode_attention(query, cache, table, lengths, use_triton=True)

        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)

    @needs_gpu
    def test_kernel_handles_a_single_token_sequence(self):
        device = torch.device("cuda")
        query, cache, table, lengths = build_case(device, [[2]], [0])

        expected = batched_decode_attention(query, cache, table, lengths, use_triton=False)
        actual = batched_decode_attention(query, cache, table, lengths, use_triton=True)

        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)


class DecodeTableTest(unittest.TestCase):
    def test_shorter_tables_are_padded_and_lengths_count_the_new_token(self):
        table, lengths = build_decode_table(
            [[5, 6, 7], [4]], [33, 2], torch.device("cpu")
        )

        self.assertEqual(table.tolist(), [[5, 6, 7], [4, 0, 0]])
        self.assertEqual(lengths.tolist(), [34, 3])

    def test_fallback_reads_only_real_blocks_of_a_padded_table(self):
        cache = torch.zeros(2, 4, 2, 1, 2)
        cache[0, 3, 0, 0] = torch.tensor([1.0, 0.0])
        cache[1, 3, 0, 0] = torch.tensor([7.0, 8.0])
        # Block 0 is padding here, so its contents must not reach the result.
        cache[0, 0] = 99.0
        cache[1, 0] = 99.0
        table, lengths = build_decode_table([[3, 0]], [0], torch.device("cpu"))
        query = torch.ones(1, 1, 2)

        result = batched_decode_attention(query, cache, table, lengths, use_triton=False)

        torch.testing.assert_close(result, torch.tensor([[[7.0, 8.0]]]))


if __name__ == "__main__":
    unittest.main()
