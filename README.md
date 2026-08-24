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
same prompt prefix can point to the same blocks. If one sequence needs to change
a shared block, the engine copies that block first. This is called copy on
write.

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
| Mean latency | 17.60 s | 13.26 s | -25% |
| Mean TTFT | 6.90 s | 4.84 s | -30% |
| Peak blocks held | 166 | 58 | -65% |
| Peak KV slots (logical) | 2656 | 2368 | -11% |
| KV slots reserved but unused | 8% | 0% | gone |
| Prefix blocks reused | 0 | 186 | |
| Wall clock | 23.89 s | 22.61 s | 1.06x |

Continuous batching puts 33% more sequences in each model step. Both engines
generate the same 1256 output tokens, but miniVLLM uses 72 steps instead of 96.
Short requests can also finish without waiting for every long request in their
batch.

The basic engine holds 166 blocks at its peak. These are the 2656 slots it
reserved, and none of them are shared. miniVLLM holds 58 blocks at its peak for
the same work, which is 2.9 times less. Its block tables need 148 blocks in total,
but only 58 physical blocks are used because prompt blocks are shared.

The total running time changes by only 1.06 times. In `PagedGPT2.decode`, the
projection calculations are batched but attention is not. Each sequence has a
different block table, so a step with 32 sequences makes 32 attention calls one
after another in Python. The number of steps falls by 1.33 times, but each step
does 1.33 times more work. A fused paged attention CUDA kernel can batch these
attention calls. This project does not have that kernel, so latency improves but
the total throughput stays almost the same.

### Tight KV pool (512 tokens)

This uses the same work with the pool reduced to 90 MiB. The test checks whether
each engine can finish all requests.

| | Static + reserved KV | miniVLLM |
| --- | --- | --- |
| Status | out of KV blocks | completed all 64 |
| Wall clock | | 23.47 s |
| Mean latency | | 12.57 s |
| Peak blocks held | | 32 of 32 |
| Peak KV slots (logical) | | 1488 |
| Peak concurrent sequences | | 23 |
| Preemptions | | 17 |
| Prefix blocks reused | | 237 |

The basic engine cannot start its first batch. Reserving `prompt + max_tokens`
for 32 sequences needs more memory than this pool has. miniVLLM finishes all 64
requests in a pool that is one thirty-second of the larger pool. It allocates
blocks when needed, shares prompt blocks, and removes sequences for recomputation
17 times when the pool is full. Its mean latency is lower than in the larger
pool because fewer sequences run at the same time, so each sequence spends less
time waiting.

The two block values count different things. Peak KV slots adds the size of every
sequence's block table. A block shared by four sequences is counted four times,
which is why the value can be 1488 when the physical pool has only 512 token
slots. Peak blocks held counts real blocks. A value of 32 out of 32 means the
engine used the whole pool.

## Layout

```
minivllm/
   block_manager.py   blocks, reference counts, prefix hashes, block copying
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
