"""Overlap-scheduling evidence: serialized vs. overlapped host/device execution.

    python -m self_improving_diffusion.jax_backend.overlap_cli --out reports/overlap_scheduling.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import jax

from .model import TinyEpsilonModel
from .overlap import run_overlapped, run_serialized
from .sampler import make_schedule
from .serving import CompiledVariantCache


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--chunk-steps", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("reports/overlap_scheduling.json"))
    args = parser.parse_args(argv)

    params = TinyEpsilonModel.init(jax.random.PRNGKey(0))
    schedule = make_schedule(16)

    results = []
    for prep_ms in (0.0, 1.0, 5.0, 20.0):
        prep_s = prep_ms / 1000.0
        cache_serial = CompiledVariantCache(params, schedule, max_variants=2)
        cache_overlap = CompiledVariantCache(params, schedule, max_variants=2)

        serialized = run_serialized(
            cache_serial, args.num_batches, args.batch_size, args.chunk_steps, start_step=15, simulated_host_prep_s=prep_s
        )
        overlapped = run_overlapped(
            cache_overlap, args.num_batches, args.batch_size, args.chunk_steps, start_step=15, simulated_host_prep_s=prep_s
        )
        results.append(
            {
                "simulated_host_prep_ms": prep_ms,
                "serialized": asdict(serialized),
                "overlapped": asdict(overlapped),
                "speedup": serialized.total_wall_time_s / overlapped.total_wall_time_s,
            }
        )

    output = {
        "note": (
            "simulated_host_prep_ms is a labeled time.sleep() stand-in for real "
            "host-side preprocessing (e.g. a text encoder); TinyEpsilonModel's "
            "actual device compute has no such step today. This measures the "
            "overlap *mechanism* via JAX's real asynchronous dispatch, not a "
            "claim about this model's true preprocessing cost."
        ),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(f"wrote {args.out}")
    for row in results:
        print(
            f"prep={row['simulated_host_prep_ms']:>5.1f}ms: "
            f"serialized={row['serialized']['total_wall_time_s']:.4f}s "
            f"overlapped={row['overlapped']['total_wall_time_s']:.4f}s "
            f"speedup={row['speedup']:.2f}x"
        )


if __name__ == "__main__":
    main()
