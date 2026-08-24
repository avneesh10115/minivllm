"""Runs miniVLLM through a small FastAPI server."""

import argparse
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from .config import EngineConfig
from .engine import LLMEngine
from .sequence import SamplingParams


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 32
    temperature: float = 0.0
    top_p: float = 1.0


class Server:
    def __init__(self, config: EngineConfig) -> None:
        self.engine = LLMEngine(config)
        self.waiters: dict[int, asyncio.Future] = {}

    async def loop(self) -> None:
        while True:
            if not self.engine.scheduler.has_work:
                await asyncio.sleep(0.005)
                continue
            for seq in self.engine.step():
                future = self.waiters.pop(seq.seq_id, None)
                if future is not None and not future.done():
                    future.set_result(seq)
            await asyncio.sleep(0)

    async def generate(self, request: GenerateRequest):
        params = SamplingParams(
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
        )
        seq_id = self.engine.add_request(request.prompt, params)
        future = asyncio.get_running_loop().create_future()
        self.waiters[seq_id] = future
        return await future


def build_app(config: EngineConfig) -> FastAPI:
    server = Server(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(server.loop())
        yield
        task.cancel()

    app = FastAPI(title="miniVLLM", lifespan=lifespan)

    @app.post("/generate")
    async def generate(request: GenerateRequest) -> dict:
        seq = await server.generate(request)
        return {
            "text": server.engine.runner.decode(seq.output_token_ids),
            "num_output_tokens": len(seq.output_token_ids),
            "ttft_s": seq.ttft(),
            "latency_s": seq.latency(),
        }

    @app.get("/metrics")
    async def metrics() -> dict:
        engine = server.engine
        manager = engine.block_manager
        return {
            "running": len(engine.scheduler.running),
            "waiting": len(engine.scheduler.waiting),
            "kv_utilization": manager.utilization(),
            "kv_fragmentation": manager.fragmentation(),
            "prefix_cache_hits": manager.prefix_hits,
            "prefix_cache_misses": manager.prefix_misses,
            "preemptions": engine.scheduler.num_preemptions,
            "peak_running": engine.stats.peak_running,
        }

    return app


def main() -> None:
    import uvicorn

    defaults = EngineConfig()
    parser = argparse.ArgumentParser(description="Serve a GPT-2 model with miniVLLM")
    parser.add_argument("--model", default=defaults.model)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-blocks", type=int, default=defaults.num_blocks)
    parser.add_argument("--block-size", type=int, default=defaults.block_size)
    parser.add_argument("--max-num-seqs", type=int, default=defaults.max_num_seqs)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = EngineConfig(
        model=args.model,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        max_num_seqs=args.max_num_seqs,
        device=args.device,
    )
    uvicorn.run(build_app(config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
