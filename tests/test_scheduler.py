import unittest

from minivllm.block_manager import BlockManager
from minivllm.config import EngineConfig
from minivllm.scheduler import Scheduler
from minivllm.sequence import SamplingParams, Sequence, SequenceStatus


class SchedulerTest(unittest.TestCase):
    def test_preempted_sequence_is_queued_for_recompute(self):
        config = EngineConfig(
            block_size=2,
            num_blocks=1,
            max_num_seqs=1,
            max_num_batched_tokens=4,
        )
        manager = BlockManager(num_blocks=1, block_size=2)
        scheduler = Scheduler(config, manager)
        sequence = Sequence(1, [10, 20], SamplingParams(max_tokens=2))
        scheduler.add(sequence)

        prefill = scheduler.schedule()
        self.assertEqual(prefill.scheduled, [sequence])
        self.assertEqual(sequence.num_computed_tokens, 2)

        decode = scheduler.schedule()

        self.assertEqual(decode.preempted, [sequence])
        self.assertEqual(sequence.status, SequenceStatus.WAITING)
        self.assertEqual(sequence.num_computed_tokens, 0)
        self.assertNotIn(sequence.seq_id, manager.tables)
        self.assertEqual(list(scheduler.waiting), [sequence])

        recompute = scheduler.schedule()
        self.assertEqual(recompute.scheduled, [sequence])
        self.assertEqual(sequence.status, SequenceStatus.RUNNING)
        self.assertEqual(sequence.num_computed_tokens, 2)


if __name__ == "__main__":
    unittest.main()