"""Runs requests through the scheduler and model."""

import time
from dataclasses import dataclass

from .block_manager import BlockManager
from .config import EngineConfig
from .scheduler import Scheduler
from .sequence import SamplingParams, Sequence


@dataclass
class RequestOutput:
    seq_id: int
    text: str
    token_ids: list[int]
    ttft: float
    latency: float


@dataclass
class EngineStats:
    peak_running: int = 0
    peak_kv_slots: int = 0
    peak_blocks_used: int = 0
    prefill_steps: int = 0
    decode_steps: int = 0
    generated_tokens: int = 0


class LLMEngine:
    def __init__(self, config: EngineConfig | None = None, runner=None) -> None:
        self.config = config or EngineConfig()
        self.block_manager = BlockManager(
            self.config.num_blocks,
            self.config.block_size,
            self.config.enable_prefix_caching,
        )
        self.scheduler = Scheduler(self.config, self.block_manager)
        self.stats = EngineStats()
        self.next_id = 0
        self.model_runner = runner

    @property
    def runner(self):
        if self.model_runner is None:
            from .model_runner import ModelRunner

            self.model_runner = ModelRunner(self.config)
        return self.model_runner

    def add_request(self, prompt: str, params: SamplingParams | None = None) -> int:
        return self.add_tokens(self.runner.encode(prompt), params)

    def add_tokens(self, token_ids: list[int], params: SamplingParams | None = None) -> int:
        params = params or SamplingParams()
        budget = self.config.max_model_len - len(token_ids)
        if budget <= 0:
            raise ValueError(f"prompt of {len(token_ids)} tokens fills the context window")
        params.max_tokens = min(params.max_tokens, budget)
        seq_id = self.next_id
        self.next_id += 1
        self.scheduler.add(Sequence(seq_id, list(token_ids), params))
        return seq_id

    def step(self) -> list[Sequence]:
        batch = self.scheduler.schedule()
        if batch.is_empty:
            return []
        block_tables = {
            seq.seq_id: self.block_manager.tables[seq.seq_id].blocks
            for seq in batch.scheduled
        }
        next_tokens = self.runner.execute(
            batch.scheduled, block_tables, batch.is_prefill_batch
        )

        if batch.is_prefill_batch:
            self.stats.prefill_steps += 1
        else:
            self.stats.decode_steps += 1
        self.stats.peak_running = max(self.stats.peak_running, len(self.scheduler.running))
        # Shared blocks are counted once for every sequence using them here.
        self.stats.peak_kv_slots = max(
            self.stats.peak_kv_slots,
            sum(table.num_slots for table in self.block_manager.tables.values()),
        )
        self.stats.peak_blocks_used = max(
            self.stats.peak_blocks_used, self.block_manager.blocks_in_use
        )
        self.stats.generated_tokens += len(batch.scheduled)

        finished: list[Sequence] = []
        for seq, token_id in zip(batch.scheduled, next_tokens):
            seq.append_token(token_id)
            if seq.is_finished():
                seq.finish_time = time.monotonic()
                finished.append(seq)
        for seq in finished:
            self.scheduler.finish(seq)
        return finished

    def run(self) -> list[RequestOutput]:
        results: list[RequestOutput] = []
        while self.scheduler.has_work:
            for seq in self.step():
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

    def generate(self, prompts: list[str], params: SamplingParams | None = None) -> list[str]:
        for prompt in prompts:
            self.add_request(prompt, params)
        outputs = self.run()
        return [output.text for output in sorted(outputs, key=lambda output: output.seq_id)]
