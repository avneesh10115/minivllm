# miniVLLM

miniVLLM is a small LLM inference engine. It uses a paged KV cache and continuous
batching, which are two methods used by vLLM. It loads real GPT-2 weights from
Hugging Face. The model forward pass reads from and writes to the paged cache.

The project also has a basic engine for comparison. This engine reserves
`prompt + max_tokens` slots for every sequence and runs a fixed batch until all
sequences in it are finished.

## The problem

A model stores keys and values in a KV cache while it generates text. The cache
grows by one token at a time. If every sequence gets one continuous area of
memory, that area must be large enough for the longest possible output. This
causes two problems:

1. Some reserved memory is never used. If a request asks for 128 tokens but
   stops at 20, the other 108 slots stay empty until the request is removed.
2. Short requests wait for long requests when a whole batch has to finish before
   another request can start.

## The approach

### Paged KV cache

The cache is split into blocks of a fixed size. The default block size is 16
tokens. Each sequence stores a list of block ids called a block table. It gets
new blocks only when it needs them, so at most one partly filled block is wasted
for each sequence.

### Sharing prompt blocks

The engine calculates a hash for every full prompt block. Sequences with the
same prompt prefix can point to the same blocks. Only full prompt blocks are
shared. Generated tokens use a new block after a full prompt block, so shared
blocks are never changed.

This sharing reduces the physical memory used by the KV cache, so a prompt that
many requests have in common costs one copy of blocks instead of one copy per
request.

### Continuous batching

The engine builds the batch again after every step. A finished sequence leaves
right away, and a waiting sequence can take its place.

### Recomputing the cache

If there are no free blocks, the engine removes the newest sequences from the
running batch and frees their cache blocks. It keeps the tokens they already
generated. When a removed sequence runs again, the engine rebuilds its cache
from its prompt and output tokens.

## Results

`gpt2-large`, fp16, one T4, 64 requests sharing a long system prompt with output
lengths from 8 to 96 tokens. miniVLLM uses `max_num_seqs=32`, and the basic
engine uses a static batch size of 32.

### Roomy KV pool (16,384 tokens)

Both engines fit in this pool. The results show how their scheduling differs.

| Metric | Static + reserved KV | miniVLLM | Change |
| --- | --- | --- | --- |
| Model steps | 96 | 72 | -25% |
| Tokens per step | 13.08 | 17.44 | +33% |
| Mean latency | 4.21 s | 2.03 s | -52% |
| Mean TTFT | 2.65 s | 1.29 s | -51% |
| Peak blocks held | 166 | 58 | -65% |
| Peak KV slots (logical) | 2656 | 2368 | -11% |
| KV slots reserved but unused | 8% | 0% | gone |
| Prefix blocks reused | 0 | 186 | |
| Wall clock | 5.12 s | 3.11 s | 1.65x |

Continuous batching puts 33% more sequences in each model step. Both engines
generate the same 1256 output tokens, but miniVLLM uses 72 steps instead of 96.
Short requests can also finish without waiting for every long request in their
batch.

The basic engine holds 166 blocks at its peak. These are the 2656 slots it
reserved, and none of them are shared. miniVLLM holds 58 blocks at its peak for
the same work, which is 2.9 times less. Its block tables need 148 blocks in total,
but only 58 physical blocks are used because prompt blocks are shared.

### Tight KV pool (512 tokens)

This uses the same work with the pool reduced to 90 MiB. The test checks whether
each engine can finish all requests.

| | Static + reserved KV | miniVLLM |
| --- | --- | --- |
| Status | out of KV blocks | completed all 64 |
| Wall clock | | 4.30 s |
| Mean latency | | 2.37 s |
| Peak blocks held | | 32 of 32 |
| Peak KV slots (logical) | | 1488 |
| Peak concurrent sequences | | 23 |
| Preemptions | | 17 |
| Prefix blocks reused | | 234 |

The basic engine cannot start its first batch. Reserving `prompt + max_tokens`
for 32 sequences needs more memory than this pool has. miniVLLM finishes all 64
requests in a pool that is one thirty-second of the larger pool. It allocates
blocks when needed, shares prompt blocks, and removes sequences for recomputation
17 times when the pool is full. Every request still finishes in a pool the basic
engine cannot even start in, and the recomputation costs 117 steps instead of 72
with a mean latency of 2.37 s instead of 2.03 s.

The two block values count different things. Peak KV slots adds the size of every
sequence's block table. A block shared by four sequences is counted four times,
which is why the value can be 1488 when the physical pool has only 512 token
slots. Peak blocks held counts real blocks. A value of 32 out of 32 means the
engine used the whole pool.

### The paged attention kernel

Decode used to call attention once per sequence. Every sequence has its own
block table, so a step with 32 sequences ran 32 separate calls in Python, and
that happened again for each of the 36 layers. Each call also gathered the whole
block table out of the cache before slicing it back down.

`minivllm/kernels.py` has a Triton kernel that does the batch in one launch. It
runs one program per sequence and head, follows the block table on the GPU, and
keeps a running softmax so it never holds every score at once. Passing
`--no-triton` switches back to the old path, which is how the numbers below were
measured.

| Metric | Per-sequence loop | Triton kernel |
| --- | --- | --- |
| Wall clock | 16.34 s | 3.11 s |
| Output tokens per second | 76.9 | 404.3 |
| Mean latency | 9.55 s | 2.03 s |
| Model steps | 72 | 72 |

The step count, the block counts and the 1256 generated tokens are the same in
both runs, because the kernel only changes how attention is computed. Generated
text is identical too. The old path issued 82,944 small attention calls from
Python for this workload, so most of the run went on call overhead rather than
arithmetic. Collapsing them into one launch per step is where the speedup comes
from.

The kernel needs a GPU, so `tests/test_paged_kernel.py` skips its two kernel
tests on a machine without one. `bench/results/gpt2-large-t4-no-kernel.json` is
the `--no-triton` run of the same workload.

## Layout

```
minivllm/
   block_manager.py   blocks, reference counts and prefix hashes
  scheduler.py       continuous batching, admission, preemption
  attention.py       paged KV read/write and attention
  model.py           GPT-2 forward pass over the paged cache
   model_runner.py    text encoding and token sampling
   engine.py          runs the model one step at a time
   baseline.py        fixed batches with reserved KV memory
   server.py          FastAPI server
bench/benchmark.py   the comparison above
```

## Running it

```bash
pip install -e .

python bench/benchmark.py --model gpt2 --num-requests 16
```

On a GPU:

```bash
python bench/benchmark.py --model gpt2-large --device cuda --dtype float16 \
    --num-requests 64 --max-num-seqs 32 --static-batch-size 32 --num-blocks 1024
```

The tight pool test uses the measured peak of the basic engine to choose its
size. You do not need to set this value by hand.

To serve:

```bash
pip install -e ".[serve]"
minivllm-serve --model gpt2
curl -s localhost:8000/generate -H 'content-type: application/json' \
     -d '{"prompt": "The capital of France is", "max_tokens": 16}'
```

## Tests

```bash
python -m pytest tests
```

The tests cover block sharing and reference counts, reads and writes across two
blocks that are not next to each other, and preemption followed by a recompute.
They run on CPU and do not download model weights.
