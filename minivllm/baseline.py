"""A basic engine used to compare results with miniVLLM."""

import time
from dataclasses import dataclass

from .block_manager import BlockManager, OutOfBlocksError
from .config import EngineConfig
from .engine import RequestOutput
from .sequence import SamplingParams, Sequence


@dataclass
class BaselineStats:
    peak_running: int = 0
    peak_kv_slots: int = 0
    peak_blocks_used: int = 0
    prefill_steps: int = 0
    decode_steps: int = 0
    generated_tokens: int = 0
    reserved_slots: int = 0
    used_slots: int = 0

    @property
    def reservation_waste(self) -> float:
        if self.reserved_slots == 0:
            return 0.0
        return 1.0 - self.used_slots / self.reserved_slots


class StaticBatchEngine:
    def __init__(self, config: EngineConfig, runner, batch_size: int = 4) -> None:
        self.config = config
        self.runner = runner
        self.batch_size = batch_size
        # The basic engine does not share prompt blocks.
        self.block_manager = BlockManager(
            config.num_blocks, config.block_size, enable_prefix_caching=False
        )
        self.stats = BaselineStats()
        self.pending: list[Sequence] = []
        self.next_id = 0

    def add_tokens(self, token_ids: list[int], params: SamplingParams | None = None) -> int:
        params = params or SamplingParams()
        seq_id = self.next_id
        self.next_id += 1
        self.pending.append(Sequence(seq_id, list(token_ids), params))
        return seq_id

    def add_request(self, prompt: str, params: SamplingParams | None = None) -> int:
        return self.add_tokens(self.runner.encode(prompt), params)

    def run(self) -> list[RequestOutput]:
        results: list[RequestOutput] = []
        for start in range(0, len(self.pending), self.batch_size):
            batch = self.pending[start : start + self.batch_size]
            results.extend(self.run_batch(batch))
        self.pending.clear()
        return results

    def run_batch(self, batch: list[Sequence]) -> list[RequestOutput]:
        for seq in batch:
            reserve = seq.num_tokens + seq.params.max_tokens
            try:
                self.block_manager.allocate(seq.seq_id, seq.token_ids, reserve=reserve)
            except OutOfBlocksError as exc:
                raise OutOfBlocksError(
                    f"batch of {len(batch)} needs {reserve} reserved slots per sequence; "
                    f"the pool holds {self.config.kv_capacity_tokens}. "
                    "Lower batch_size or raise num_blocks."
                ) from exc
            self.stats.reserved_slots += self.block_manager.tables[seq.seq_id].num_slots

        self.stats.peak_kv_slots = max(
            self.stats.peak_kv_slots,
            sum(self.block_manager.tables[seq.seq_id].num_slots for seq in batch),
        )
        self.stats.peak_blocks_used = max(
            self.stats.peak_blocks_used, self.block_manager.blocks_in_use
        )
        block_tables = {
            seq.seq_id: self.block_manager.tables[seq.seq_id].blocks for seq in batch
        }
        for token_id, seq in zip(
            self.runner.execute(batch, block_tables, is_prefill=True), batch
        ):
            seq.append_token(token_id)
        self.stats.prefill_steps += 1
        self.stats.generated_tokens += len(batch)
        self.stats.peak_running = max(self.stats.peak_running, len(batch))

        # Keep running the batch until every sequence is finished.
        active = [seq for seq in batch if not seq.is_finished()]
        while active:
            for seq in active:
                self.block_manager.append_token(seq.seq_id)
            for token_id, seq in zip(
                self.runner.execute(active, block_tables, is_prefill=False), active
            ):
                seq.append_token(token_id)
            self.stats.decode_steps += 1
            self.stats.generated_tokens += len(active)
            active = [seq for seq in active if not seq.is_finished()]

        results = []
        for seq in batch:
            seq.finish_time = time.monotonic()
            self.stats.used_slots += seq.num_tokens
            self.block_manager.free(seq.seq_id)
            results.append(
                RequestOutput(
                    seq_id=seq.seq_id,
                    text=self.runner.decode(seq.output_token_ids),
                    token_ids=list(seq.output_token_ids),
                    ttft=seq.ttft(),
                    latency=seq.latency(),
                )
            )
        return results
