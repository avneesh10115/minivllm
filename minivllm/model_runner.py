"""Loads the model and chooses the next token."""

import torch

from .config import EngineConfig
from .model import PagedGPT2
from .sequence import Sequence


class ModelRunner:
    def __init__(self, config: EngineConfig) -> None:
        from transformers import AutoTokenizer

        self.config = config
        self.device = config.resolve_device()
        self.tokenizer = AutoTokenizer.from_pretrained(config.model)
        self.model = PagedGPT2.from_pretrained(
            config.model,
            num_blocks=config.num_blocks,
            block_size=config.block_size,
            device=self.device,
            dtype=config.resolve_dtype(),
        )
        self.eos_token_id = self.tokenizer.eos_token_id

    def encode(self, prompt: str) -> list[int]:
        return self.tokenizer.encode(prompt)

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def execute(
        self, batch: list[Sequence], block_tables: dict[int, list[int]], is_prefill: bool
    ) -> list[int]:
        if is_prefill:
            logits = torch.stack(
                [self.model.prefill(seq.token_ids, block_tables[seq.seq_id]) for seq in batch]
            )
        else:
            logits = self.model.decode(
                token_ids=[seq.output_token_ids[-1] for seq in batch],
                block_tables=[block_tables[seq.seq_id] for seq in batch],
                cache_lens=[seq.num_tokens - 1 for seq in batch],
            )
        return [self.sample(logits[index], seq) for index, seq in enumerate(batch)]

    def sample(self, logits: torch.Tensor, seq: Sequence) -> int:
        temperature = seq.params.temperature
        if temperature <= 0:
            return int(logits.argmax(dim=-1).item())
        probabilities = torch.softmax(logits.float() / temperature, dim=-1)
        if seq.params.top_p < 1.0:
            probabilities = self.filter_top_p(probabilities, seq.params.top_p)
        return int(torch.multinomial(probabilities, num_samples=1).item())

    @staticmethod
    def filter_top_p(probabilities: torch.Tensor, top_p: float) -> torch.Tensor:
        sorted_probabilities, sorted_indexes = probabilities.sort(descending=True)
        previous_total = sorted_probabilities.cumsum(dim=-1) - sorted_probabilities
        sorted_probabilities[previous_total >= top_p] = 0.0
        filtered = torch.zeros_like(probabilities)
        filtered[sorted_indexes] = sorted_probabilities
        return filtered / filtered.sum()
