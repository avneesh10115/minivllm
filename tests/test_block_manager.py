import unittest

from minivllm.block_manager import BlockManager, OutOfBlocksError


class BlockManagerTest(unittest.TestCase):
    def test_full_prompt_blocks_are_shared_and_released(self):
        manager = BlockManager(num_blocks=3, block_size=2)

        first = manager.allocate(1, [10, 20, 30])
        second = manager.allocate(2, [10, 20, 40])

        self.assertEqual(first.blocks[0], second.blocks[0])
        self.assertNotEqual(first.blocks[1], second.blocks[1])
        self.assertEqual(manager.allocator.block(first.blocks[0]).ref_count, 2)
        self.assertEqual(manager.prefix_hits, 1)

        manager.free(1)
        self.assertEqual(manager.allocator.block(second.blocks[0]).ref_count, 1)
        manager.free(2)
        self.assertEqual(manager.allocator.num_free, 3)

    def test_failed_allocation_releases_earlier_blocks(self):
        manager = BlockManager(num_blocks=1, block_size=2)

        with self.assertRaises(OutOfBlocksError):
            manager.allocate(1, [10, 20, 30])

        self.assertNotIn(1, manager.tables)
        self.assertEqual(manager.allocator.num_free, 1)

    def test_stale_prefix_entry_does_not_reuse_reallocated_block(self):
        manager = BlockManager(num_blocks=2, block_size=2)

        old_table = manager.allocate(1, [10, 20])
        old_block = old_table.blocks[0]
        manager.free(1)

        different_table = manager.allocate(2, [30])
        self.assertEqual(different_table.blocks[0], old_block)

        repeated_table = manager.allocate(3, [10, 20])

        self.assertNotEqual(repeated_table.blocks[0], different_table.blocks[0])
        self.assertEqual(manager.prefix_hits, 0)


if __name__ == "__main__":
    unittest.main()