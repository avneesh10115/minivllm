"""Chooses which requests run in each model step."""

from collections import deque
from dataclasses import dataclass, field

from .block_manager import BlockManager, OutOfBlocksError
from .config import EngineConfig
from .sequence import Sequence, SequenceStatus


@dataclass
class SchedulerOutput:
    scheduled: list[Sequence] = field(default_factory=list)
    preempted: list[Sequence] = field(default_factory=list)
    is_prefill_batch: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.scheduled


class Scheduler:
    def __init__(self, config: EngineConfig, block_manager: BlockManager) -> None:
        self.config = config
        self.block_manager = block_manager
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.num_preemptions = 0

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def schedule(self) -> SchedulerOutput:
        result = self.schedule_new_requests()
        if not result.is_empty:
            return result
        return self.schedule_running_requests()

    def schedule_new_requests(self) -> SchedulerOutput:
        result = SchedulerOutput(is_prefill_batch=True)
        budget = self.config.max_num_batched_tokens
        while (
            self.waiting
            and len(self.running) + len(result.scheduled) < self.config.max_num_seqs
        ):
            seq = self.waiting[0]
            tokens_to_process = seq.num_tokens - seq.num_computed_tokens
            if tokens_to_process > budget and result.scheduled:
                break
            if not self.block_manager.can_allocate(seq.num_tokens):
                break
            self.waiting.popleft()
            # A removed request also needs to rebuild the cache for its output.
            self.block_manager.allocate(seq.seq_id, seq.token_ids)
            seq.status = SequenceStatus.RUNNING
            seq.num_computed_tokens = seq.num_tokens
            budget -= tokens_to_process
            result.scheduled.append(seq)
        self.running.extend(result.scheduled)
        return result

    def schedule_running_requests(self) -> SchedulerOutput:
        result = SchedulerOutput(is_prefill_batch=False)
        still_running: list[Sequence] = []
        pending = deque(self.running)
        while pending:
            seq = pending.popleft()
            gave_up = False
            while not self.try_to_add_token_slot(seq):
                # Remove the newest request first. If there is no other request,
                # this request has to free its own blocks.
                if pending:
                    victim = pending.pop()
                elif still_running:
                    victim = still_running.pop()
                else:
                    victim = seq
                self.remove_from_batch(victim)
                result.preempted.append(victim)
                if victim is seq:
                    gave_up = True
                    break
            if not gave_up:
                still_running.append(seq)
        self.running = still_running
        result.scheduled = list(still_running)
        return result

    def try_to_add_token_slot(self, seq: Sequence) -> bool:
        try:
            self.block_manager.append_token(seq.seq_id)
        except OutOfBlocksError:
            return False
        return True

    def remove_from_batch(self, seq: Sequence) -> None:
        # For short sequences, rebuilding is cheaper than moving the whole cache
        # to the CPU and back.
        self.num_preemptions += 1
        self.block_manager.free(seq.seq_id)
        seq.reset_for_recompute()
        self.waiting.appendleft(seq)

    def finish(self, seq: Sequence) -> None:
        seq.status = SequenceStatus.FINISHED
        self.block_manager.free(seq.seq_id)
        if seq in self.running:
            self.running.remove(seq)
