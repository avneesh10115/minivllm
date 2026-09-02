import unittest

import torch

from minivllm.attention import read_kv, write_kv


class PagedCacheTest(unittest.TestCase):
    def test_write_and_read_across_non_contiguous_blocks(self):
        cache = torch.zeros((2, 3, 2, 1, 2), dtype=torch.float32)
        keys = torch.tensor([[[1, 2]], [[3, 4]], [[5, 6]]])
        values = keys + 10

        write_kv(cache, [2, 0], start_pos=1, keys=keys, values=values)
        saved_keys, saved_values = read_kv(cache, [2, 0], length=4)
        expected_keys = torch.cat((torch.zeros((1, 1, 2)), keys.float()))
        expected_values = torch.cat((torch.zeros((1, 1, 2)), values.float()))

        self.assertTrue(torch.equal(saved_keys, expected_keys))
        self.assertTrue(torch.equal(saved_values, expected_values))
