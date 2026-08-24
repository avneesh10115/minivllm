import enum
import time
from dataclasses import dataclass, field


class SequenceStatus(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class SamplingParams:
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    stop_token_ids: list[int] = field(default_factory=list)


@dataclass
class Sequence:
    seq_id: int
    prompt_token_ids: list[int]
    params: SamplingParams
    status: SequenceStatus = SequenceStatus.WAITING
    output_token_ids: list[int] = field(default_factory=list)
    arrival_time: float = field(default_factory=time.monotonic)
    first_token_time: float | None = None
    finish_time: float | None = None
    # Number of tokens already stored in the KV cache.
    num_computed_tokens: int = 0

    @property
    def token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def num_tokens(self) -> int:
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    def append_token(self, token_id: int) -> None:
        if self.first_token_time is None:
            self.first_token_time = time.monotonic()
        self.output_token_ids.append(token_id)

    def is_finished(self) -> bool:
        if self.status is SequenceStatus.FINISHED:
            return True
        if len(self.output_token_ids) >= self.params.max_tokens:
            return True
        if not self.output_token_ids:
            return False
        return self.output_token_ids[-1] in self.params.stop_token_ids

    def reset_for_recompute(self) -> None:
        # Generated tokens stay in the list. The cache is rebuilt next time.
        self.num_computed_tokens = 0
        self.status = SequenceStatus.WAITING

    def ttft(self) -> float | None:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    def latency(self) -> float | None:
        if self.finish_time is None:
            return None
        return self.finish_time - self.arrival_time
