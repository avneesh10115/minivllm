from dataclasses import dataclass

import torch


@dataclass
class EngineConfig:
    model: str = "gpt2"
    device: str | None = None
    dtype: str = "float32"

    # KV cache settings shared by all requests.
    block_size: int = 16
    num_blocks: int = 256
    enable_prefix_caching: bool = True

    # Limits used when the scheduler builds a batch.
    max_num_seqs: int = 16
    max_num_batched_tokens: int = 2048
    max_model_len: int = 1024

    @property
    def kv_capacity_tokens(self) -> int:
        return self.num_blocks * self.block_size

    def resolve_device(self) -> torch.device:
        if self.device:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def resolve_dtype(self) -> torch.dtype:
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.dtype]
