"""Stores the KV cache in small blocks instead of one large area."""

import hashlib
from dataclasses import dataclass, field


class OutOfBlocksError(RuntimeError):
    pass


@dataclass
class PhysicalBlock:
    block_id: int
    ref_count: int = 0
    # This is set after the block is full.
    content_hash: str | None = None


class BlockAllocator:
    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.num_blocks = num_blocks
        self.blocks = [PhysicalBlock(block_id) for block_id in range(num_blocks)]
        self.free_blocks: list[int] = list(reversed(range(num_blocks)))

    @property
    def num_free(self) -> int:
        return len(self.free_blocks)

    def block(self, block_id: int) -> PhysicalBlock:
        return self.blocks[block_id]

    def allocate(self) -> PhysicalBlock:
        if not self.free_blocks:
            raise OutOfBlocksError("no free KV blocks")
        block = self.blocks[self.free_blocks.pop()]
        block.ref_count = 1
        block.content_hash = None
        return block

    def add_reference(self, block_id: int) -> None:
        self.blocks[block_id].ref_count += 1

    def free(self, block_id: int) -> None:
        block = self.blocks[block_id]
        if block.ref_count <= 0:
            raise RuntimeError(f"double free of block {block_id}")
        block.ref_count -= 1
        if block.ref_count == 0:
            block.content_hash = None
            self.free_blocks.append(block_id)


def hash_tokens(token_ids: list[int]) -> str:
    token_bytes = ",".join(map(str, token_ids)).encode()
    return hashlib.sha256(token_bytes).hexdigest()


@dataclass
class BlockTable:
    block_size: int
    blocks: list[int] = field(default_factory=list)
    num_tokens: int = 0

    @property
    def num_slots(self) -> int:
        return len(self.blocks) * self.block_size

    def slot_index(self, position: int) -> tuple[int, int]:
        if position >= self.num_tokens:
            raise IndexError(position)
        return self.blocks[position // self.block_size], position % self.block_size


class BlockManager:
    def __init__(
        self, num_blocks: int, block_size: int = 16, enable_prefix_caching: bool = True
    ) -> None:
        self.block_size = block_size
        self.enable_prefix_caching = enable_prefix_caching
        self.allocator = BlockAllocator(num_blocks)
        self.tables: dict[int, BlockTable] = {}
        # A prompt block hash points to the block where it is stored.
        self.prefix_cache: dict[str, int] = {}
        self.prefix_hits = 0
        self.prefix_misses = 0

    def blocks_needed(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def can_allocate(self, num_tokens: int) -> bool:
        return self.blocks_needed(num_tokens) <= self.allocator.num_free

    def allocate(
        self, seq_id: int, prompt_token_ids: list[int], reserve: int | None = None
    ) -> BlockTable:
        """Give cache blocks to a prompt and reuse matching full blocks."""
        if seq_id in self.tables:
            raise KeyError(f"sequence {seq_id} already allocated")
        table = BlockTable(self.block_size)
        self.tables[seq_id] = table
        try:
            for start in range(0, len(prompt_token_ids), self.block_size):
                chunk = prompt_token_ids[start : start + self.block_size]
                if len(chunk) == self.block_size and self.enable_prefix_caching:
                    self.add_shared_block(table, prompt_token_ids[: start + self.block_size])
                else:
                    table.blocks.append(self.allocator.allocate().block_id)
                table.num_tokens += len(chunk)
            while reserve is not None and table.num_slots < reserve:
                table.blocks.append(self.allocator.allocate().block_id)
        except OutOfBlocksError:
            self.free(seq_id)
            raise
        return table

    def add_shared_block(self, table: BlockTable, prefix_tokens: list[int]) -> None:
        token_hash = hash_tokens(prefix_tokens)
        cached_block_id = self.prefix_cache.get(token_hash)
        if cached_block_id is not None:
            cached = self.allocator.block(cached_block_id)
            # A freed block is handed out again for other tokens, so the hash on
            # the block has to match too, not just the one in the cache.
            if cached.ref_count > 0 and cached.content_hash == token_hash:
                self.allocator.add_reference(cached_block_id)
                table.blocks.append(cached_block_id)
                self.prefix_hits += 1
                return

        block = self.allocator.allocate()
        block.content_hash = token_hash
        self.prefix_cache[token_hash] = block.block_id
        table.blocks.append(block.block_id)
        self.prefix_misses += 1

    def append_token(self, seq_id: int) -> int:
        table = self.tables[seq_id]
        if table.num_tokens == table.num_slots:
            table.blocks.append(self.allocator.allocate().block_id)
        table.num_tokens += 1
        return table.blocks[-1]

    def fork(self, parent_seq_id: int, child_seq_id: int) -> BlockTable:
        parent = self.tables[parent_seq_id]
        child = BlockTable(self.block_size, list(parent.blocks), parent.num_tokens)
        for block_id in child.blocks:
            self.allocator.add_reference(block_id)
        self.tables[child_seq_id] = child
        return child

    def ensure_writable(self, seq_id: int) -> int | None:
        table = self.tables[seq_id]
        if not table.blocks:
            return None
        last = table.blocks[-1]
        if self.allocator.block(last).ref_count == 1:
            return None
        new_block = self.allocator.allocate()
        table.blocks[-1] = new_block.block_id
        self.allocator.free(last)
        return new_block.block_id

    def free(self, seq_id: int) -> None:
        table = self.tables.pop(seq_id, None)
        if table is None:
            return
        for block_id in table.blocks:
            self.allocator.free(block_id)

    def utilization(self) -> float:
        used = self.allocator.num_blocks - self.allocator.num_free
        return used / self.allocator.num_blocks

    @property
    def blocks_in_use(self) -> int:
        return self.allocator.num_blocks - self.allocator.num_free

    def fragmentation(self) -> float:
        allocated_slots = sum(table.num_slots for table in self.tables.values())
        if allocated_slots == 0:
            return 0.0
        live = sum(table.num_tokens for table in self.tables.values())
        return 1.0 - live / allocated_slots
