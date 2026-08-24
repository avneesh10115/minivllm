"""Runs GPT-2 with the paged KV cache used by this project."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .attention import decode_attention, prefill_attention, read_kv, write_kv


@dataclass
class LayerWeights:
    ln1_w: torch.Tensor
    ln1_b: torch.Tensor
    qkv_w: torch.Tensor
    qkv_b: torch.Tensor
    proj_w: torch.Tensor
    proj_b: torch.Tensor
    ln2_w: torch.Tensor
    ln2_b: torch.Tensor
    fc_w: torch.Tensor
    fc_b: torch.Tensor
    fc_proj_w: torch.Tensor
    fc_proj_b: torch.Tensor


class PagedGPT2:
    def __init__(
        self,
        hf_model,
        num_blocks: int,
        block_size: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        config = hf_model.config
        self.device = torch.device(device)
        self.dtype = dtype
        self.num_layers = config.n_layer
        self.num_heads = config.n_head
        self.hidden_size = config.n_embd
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position = config.n_positions
        self.vocab_size = config.vocab_size
        self.eps = config.layer_norm_epsilon
        self.block_size = block_size
        self.num_blocks = num_blocks

        hf_model = hf_model.to(device=self.device, dtype=dtype).eval()
        transformer = hf_model.transformer
        self.wte = transformer.wte.weight.detach()
        self.wpe = transformer.wpe.weight.detach()
        self.lnf_w = transformer.ln_f.weight.detach()
        self.lnf_b = transformer.ln_f.bias.detach()
        self.layers = [self.extract_weights(block) for block in transformer.h]

        self.cache = [
            torch.zeros(
                2, num_blocks, block_size, self.num_heads, self.head_dim,
                device=self.device, dtype=dtype,
            )
            for _ in range(self.num_layers)
        ]

    @staticmethod
    def extract_weights(block) -> LayerWeights:
        # Hugging Face Conv1D stores weights as [input, output].
        return LayerWeights(
            ln1_w=block.ln_1.weight.detach(),
            ln1_b=block.ln_1.bias.detach(),
            qkv_w=block.attn.c_attn.weight.detach(),
            qkv_b=block.attn.c_attn.bias.detach(),
            proj_w=block.attn.c_proj.weight.detach(),
            proj_b=block.attn.c_proj.bias.detach(),
            ln2_w=block.ln_2.weight.detach(),
            ln2_b=block.ln_2.bias.detach(),
            fc_w=block.mlp.c_fc.weight.detach(),
            fc_b=block.mlp.c_fc.bias.detach(),
            fc_proj_w=block.mlp.c_proj.weight.detach(),
            fc_proj_b=block.mlp.c_proj.bias.detach(),
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        num_blocks: int,
        block_size: int,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "PagedGPT2":
        from transformers import GPT2LMHeadModel

        hf_model = GPT2LMHeadModel.from_pretrained(model_name)
        return cls(hf_model, num_blocks, block_size, device, dtype)

    @property
    def kv_bytes_per_block(self) -> int:
        per_layer = 2 * self.block_size * self.num_heads * self.head_dim
        return per_layer * self.num_layers * self.cache[0].element_size()

    def layer_norm(self, hidden_states, weight, bias):
        return F.layer_norm(hidden_states, (self.hidden_size,), weight, bias, self.eps)

    def run_mlp(self, hidden_states, layer: LayerWeights):
        activated = F.gelu(
            hidden_states @ layer.fc_w + layer.fc_b, approximate="tanh"
        )
        return activated @ layer.fc_proj_w + layer.fc_proj_b

    def split_heads(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states.view(
            *hidden_states.shape[:-1], self.num_heads, self.head_dim
        )

    @torch.inference_mode()
    def prefill(self, token_ids: list[int], block_table: list[int]) -> torch.Tensor:
        num_tokens = len(token_ids)
        if num_tokens > self.max_position:
            raise ValueError(f"prompt of {num_tokens} exceeds context {self.max_position}")
        ids = torch.tensor(token_ids, device=self.device)
        positions = torch.arange(num_tokens, device=self.device)
        hidden_states = self.wte[ids] + self.wpe[positions]

        for index, layer in enumerate(self.layers):
            normalized = self.layer_norm(hidden_states, layer.ln1_w, layer.ln1_b)
            combined_qkv = normalized @ layer.qkv_w + layer.qkv_b
            query_flat, key_flat, value_flat = combined_qkv.split(
                self.hidden_size, dim=-1
            )
            queries = self.split_heads(query_flat)
            keys = self.split_heads(key_flat)
            values = self.split_heads(value_flat)
            # A shared prompt block gets the same keys and values.
            write_kv(self.cache[index], block_table, 0, keys, values)
            attention_result = prefill_attention(queries, keys, values).reshape(
                num_tokens, self.hidden_size
            )
            hidden_states = (
                hidden_states + attention_result @ layer.proj_w + layer.proj_b
            )
            hidden_states = hidden_states + self.run_mlp(
                self.layer_norm(hidden_states, layer.ln2_w, layer.ln2_b), layer
            )

        hidden_states = self.layer_norm(hidden_states[-1], self.lnf_w, self.lnf_b)
        return hidden_states @ self.wte.T

    @torch.inference_mode()
    def decode(
        self,
        token_ids: list[int],
        block_tables: list[list[int]],
        cache_lens: list[int],
    ) -> torch.Tensor:
        """Runs one new token for every sequence in the batch."""
        batch_size = len(token_ids)
        ids = torch.tensor(token_ids, device=self.device)
        positions = torch.tensor(cache_lens, device=self.device)
        if int(positions.max()) >= self.max_position:
            raise ValueError(f"sequence exceeds context window {self.max_position}")
        hidden_states = self.wte[ids] + self.wpe[positions]

        for index, layer in enumerate(self.layers):
            normalized = self.layer_norm(hidden_states, layer.ln1_w, layer.ln1_b)
            combined_qkv = normalized @ layer.qkv_w + layer.qkv_b
            query_flat, key_flat, value_flat = combined_qkv.split(
                self.hidden_size, dim=-1
            )
            queries = self.split_heads(query_flat)
            keys = self.split_heads(key_flat)
            values = self.split_heads(value_flat)

            attention_result = torch.empty_like(queries)
            for batch_index in range(batch_size):
                cache = self.cache[index]
                write_kv(
                    cache,
                    block_tables[batch_index],
                    cache_lens[batch_index],
                    keys[batch_index : batch_index + 1],
                    values[batch_index : batch_index + 1],
                )
                cached_keys, cached_values = read_kv(
                    cache,
                    block_tables[batch_index],
                    cache_lens[batch_index] + 1,
                )
                attention_result[batch_index] = decode_attention(
                    queries[batch_index], cached_keys, cached_values
                )

            hidden_states = (
                hidden_states
                + attention_result.reshape(batch_size, self.hidden_size) @ layer.proj_w
                + layer.proj_b
            )
            hidden_states = hidden_states + self.run_mlp(
                self.layer_norm(hidden_states, layer.ln2_w, layer.ln2_b), layer
            )

        hidden_states = self.layer_norm(hidden_states, self.lnf_w, self.lnf_b)
        return hidden_states @ self.wte.T
