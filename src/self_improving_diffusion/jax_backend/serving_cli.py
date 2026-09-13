"""JX-05 evidence command: static batching vs chunk scheduling under a declared arrival process.

    python -m self_improving_diffusion.jax_backend.serving_cli --out reports/jx05_serving.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

from .model import TinyEpsilonModel
from .sampler import make_schedule
from .serving import (
    CompiledVariantCache,
    generate_poisson_arrivals,
    simulate_chunk_scheduling,
    simulate_static_batching,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rate-per-s", type=float, default=15.0)
    parser.add_argument("--num-requests", type=int, default=30)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-steps", type=int, default=2)
    parser.add_argument("--sla-s", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("reports/jx05_serving.json"))
    args = parser.parse_args(argv)

    params = TinyEpsilonModel.init(jax.random.PRNGKey(args.seed))
    schedule = make_schedule(args.steps)

    static_requests = generate_poisson_arrivals(args.rate_per_s, args.num_requests, args.steps, args.seed)
    chunk_requests = generate_poisson_arrivals(args.rate_per_s, args.num_requests, args.steps, args.seed)

    static_cache = CompiledVariantCache(params, schedule, max_variants=4)
    chunk_cache = CompiledVariantCache(params, schedule, max_variants=4)

    static_report = simulate_static_batching(static_cache, static_requests, args.batch_size, args.steps)
    chunk_report = simulate_chunk_scheduling(chunk_cache, chunk_requests, args.batch_size, args.chunk_steps, args.steps)

    output = {
        "arrival_process": {"distribution": "poisson", "rate_per_s": args.rate_per_s, "num_requests": args.num_requests},
        "static_batching": {
            **static_report.as_dict(args.sla_s),
            "compiled_variants": static_cache.num_cached_variants,
            "cache_hits": static_cache.hits,
            "cache_misses": static_cache.misses,
        },
        "chunk_scheduling": {
            **chunk_report.as_dict(args.sla_s),
            "compiled_variants": chunk_cache.num_cached_variants,
            "cache_hits": chunk_cache.hits,
            "cache_misses": chunk_cache.misses,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(f"wrote {args.out}")
    for policy in ("static_batching", "chunk_scheduling"):
        p = output[policy]
        print(
            f"{policy}: p50={p['latency_percentiles']['p50_s']:.4f}s "
            f"p95={p['latency_percentiles']['p95_s']:.4f}s "
            f"goodput={p['goodput_fraction_within_sla']:.2f} "
            f"throughput={p['throughput_rps']:.2f} req/s"
        )


if __name__ == "__main__":
    main()
