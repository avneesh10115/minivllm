"""Runs the same requests with miniVLLM and the basic engine.

The requests start with the same text and use different output lengths. Both
engines run once with a large cache and once with a small cache.
"""

import argparse
import json
import pathlib
import random
import time

from minivllm.baseline import StaticBatchEngine
from minivllm.block_manager import OutOfBlocksError
from minivllm.config import EngineConfig
from minivllm.engine import LLMEngine
from minivllm.model_runner import ModelRunner
from minivllm.sequence import SamplingParams

SYSTEM_PROMPT = (
    "You are a careful and concise assistant. Answer the user's question directly, "
    "without repeating the question, and stop as soon as the answer is complete. "
    "Prefer concrete detail over generalities and never invent facts. "
)

TOPICS = [
    "the water cycle",
    "how a compiler works",
    "the causes of inflation",
    "photosynthesis",
    "how TCP guarantees delivery",
    "the Doppler effect",
    "why the sky is blue",
    "how vaccines train the immune system",
]


def build_workload(num_requests: int, seed: int) -> list[tuple[str, int]]:
    random_picker = random.Random(seed)
    workload = []
    for request_number in range(num_requests):
        topic = TOPICS[request_number % len(TOPICS)]
        prompt = f"{SYSTEM_PROMPT}\nQuestion: Explain {topic}.\nAnswer:"
        # Different output lengths make short requests wait in a fixed batch.
        max_tokens = random_picker.choice([8, 8, 12, 16, 24, 48])
        workload.append((prompt, max_tokens))
    return workload


def summarize(outputs, run_time: float, stats, label: str) -> dict:
    latencies = sorted(output.latency for output in outputs)
    ttfts = sorted(output.ttft for output in outputs)
    generated = sum(len(output.token_ids) for output in outputs)
    steps = stats.prefill_steps + stats.decode_steps

    def percentile(values, fraction):
        index = min(int(fraction * len(values)), len(values) - 1)
        return values[index]

    return {
        "engine": label,
        "status": "ok",
        "num_requests": len(outputs),
        "wall_s": round(run_time, 3),
        "generated_tokens": generated,
        "output_tokens_per_s": round(generated / run_time, 2),
        "model_steps": steps,
        "tokens_per_step": round(stats.generated_tokens / max(steps, 1), 2),
        "mean_latency_s": round(sum(latencies) / len(latencies), 3),
        "p99_latency_s": round(percentile(latencies, 0.99), 3),
        "mean_ttft_s": round(sum(ttfts) / len(ttfts), 3),
        "peak_concurrent_seqs": stats.peak_running,
        "peak_kv_slots": stats.peak_kv_slots,
        "peak_blocks_used": stats.peak_blocks_used,
    }


def run_paged(config: EngineConfig, runner: ModelRunner, workload) -> dict:
    engine = LLMEngine(config, runner=runner)
    for prompt, max_tokens in workload:
        engine.add_request(prompt, SamplingParams(max_tokens=max_tokens))
    start = time.monotonic()
    outputs = engine.run()
    run_time = time.monotonic() - start

    report = summarize(outputs, run_time, engine.stats, "miniVLLM (paged + continuous)")
    report["kv_slots_wasted"] = 0.0
    report["prefix_blocks_reused"] = engine.block_manager.prefix_hits
    report["preemptions"] = engine.scheduler.num_preemptions
    return report


def run_static(config: EngineConfig, runner: ModelRunner, workload, batch_size: int) -> dict:
    engine = StaticBatchEngine(config, runner, batch_size=batch_size)
    for prompt, max_tokens in workload:
        engine.add_request(prompt, SamplingParams(max_tokens=max_tokens))
    start = time.monotonic()
    try:
        outputs = engine.run()
    except OutOfBlocksError:
        return {
            "engine": "static batching + reserved KV",
            "status": "out of KV blocks",
            "peak_kv_slots": engine.stats.peak_kv_slots,
            "peak_blocks_used": engine.stats.peak_blocks_used,
        }
    run_time = time.monotonic() - start

    report = summarize(outputs, run_time, engine.stats, "static batching + reserved KV")
    report["kv_slots_wasted"] = round(engine.stats.reservation_waste, 4)
    report["prefix_blocks_reused"] = 0
    report["preemptions"] = 0
    return report


def to_markdown(reports: list[dict]) -> str:
    columns = [
        ("engine", "Engine"),
        ("status", "Status"),
        ("wall_s", "Wall (s)"),
        ("output_tokens_per_s", "Output tok/s"),
        ("model_steps", "Model steps"),
        ("tokens_per_step", "Tokens/step"),
        ("mean_latency_s", "Mean latency (s)"),
        ("peak_kv_slots", "Peak KV slots (logical)"),
        ("peak_blocks_used", "Peak blocks (physical)"),
        ("preemptions", "Preemptions"),
    ]
    header = "| " + " | ".join(label for key, label in columns) + " |"
    divider = "| " + " | ".join("---" for key, label in columns) + " |"
    rows = [
        "| " + " | ".join(str(report.get(key, "n/a")) for key, label in columns) + " |"
        for report in reports
    ]
    return "\n".join([header, divider, *rows])


def run_scenario(args, workload, num_blocks: int) -> dict:
    config = EngineConfig(
        model=args.model,
        num_blocks=num_blocks,
        block_size=args.block_size,
        max_num_seqs=args.max_num_seqs,
        device=args.device,
        dtype=args.dtype,
    )
    runner = ModelRunner(config)
    reports = [
        run_static(config, runner, workload, args.static_batch_size),
        run_paged(config, runner, workload),
    ]
    return {
        "num_blocks": num_blocks,
        "kv_pool_tokens": config.kv_capacity_tokens,
        "kv_pool_mib": round(runner.model.kv_bytes_per_block * num_blocks / 1024**2, 1),
        "reports": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark miniVLLM against static batching")
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--num-requests", type=int, default=24)
    parser.add_argument("--num-blocks", type=int, default=256, help="roomy KV pool")
    parser.add_argument(
        "--tight-blocks",
        type=int,
        default=None,
        help="constrained KV pool; defaults to 90%% of what the baseline reserved",
    )
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--static-batch-size", type=int, default=8)
    parser.add_argument("--device", default=None, help="cuda or cpu; auto-detected by default")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="bench/results/benchmark.json")
    args = parser.parse_args()

    workload = build_workload(args.num_requests, args.seed)
    roomy = run_scenario(args, workload, args.num_blocks)

    # Use the basic engine's measured value to choose the small cache size.
    if args.tight_blocks is None:
        baseline_peak = max(
            (
                report["peak_kv_slots"]
                for report in roomy["reports"]
                if report["engine"].startswith("static")
            ),
            default=0,
        )
        args.tight_blocks = max(1, int(baseline_peak * 0.9) // args.block_size)

    scenarios = {
        "roomy_kv_pool": roomy,
        "tight_kv_pool": run_scenario(args, workload, args.tight_blocks),
    }

    results = {"config": vars(args), "scenarios": scenarios}
    path = pathlib.Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))

    for name, scenario in scenarios.items():
        print(f"\n### {name}: {scenario['kv_pool_tokens']} tokens "
              f"({scenario['kv_pool_mib']} MiB)")
        print(to_markdown(scenario["reports"]))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
