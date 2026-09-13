"""JX-02 evidence command: a trustworthy single-device baseline report.

    python -m self_improving_diffusion.jax_backend.benchmark_cli --out reports/jx02_baseline.json

Runs the same trajectory twice (cold: includes compilation; warm: reuses the
compiled executable), cross-checks the JAX result against the pure-NumPy
reference implementation, and writes a report with every phase's wall-clock
time synchronized before it's recorded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

from .benchmark import compare_against_numpy_reference, run_single_device_baseline
from .model import TinyEpsilonModel
from .sampler import make_schedule


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--out", type=Path, default=Path("reports/jx02_baseline.json"))
    args = parser.parse_args(argv)

    params = TinyEpsilonModel.init(jax.random.PRNGKey(args.seed))
    schedule = make_schedule(args.steps)

    jit_cache: dict = {}
    cold = run_single_device_baseline(params, schedule, args.seed, args.steps, "cold", jit_cache)
    warm = run_single_device_baseline(params, schedule, args.seed, args.steps, "warm", jit_cache)

    matches, max_abs_diff = compare_against_numpy_reference(params, schedule, args.seed)

    cold.numerical_match_vs_numpy = matches
    cold.max_abs_diff_vs_numpy = max_abs_diff

    bottleneck = max(
        [
            ("dispatch", warm.timings.dispatch_s),
            ("execute", warm.timings.execute_s),
            ("transfer", warm.timings.transfer_s),
            ("decode", warm.timings.decode_s),
            ("verify", warm.timings.verify_s),
        ],
        key=lambda item: item[1],
    )

    report = {
        "cold": cold.as_dict(),
        "warm": warm.as_dict(),
        "compile_overhead_ratio": (
            cold.timings.total_s() / warm.timings.total_s() if warm.timings.total_s() > 0 else None
        ),
        "warm_bottleneck_phase": bottleneck[0],
        "warm_bottleneck_seconds": bottleneck[1],
        "reference_comparison": {
            "implementation": "eager NumPy, matched weights/schedule/RNG draws, no JIT",
            "within_tolerance": matches,
            "max_abs_diff": max_abs_diff,
        },
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    print(f"cold total: {cold.timings.total_s():.4f}s, warm total: {warm.timings.total_s():.4f}s")
    print(f"warm bottleneck: {bottleneck[0]} ({bottleneck[1]:.6f}s)")
    print(f"matches NumPy reference within tolerance: {matches} (max abs diff {max_abs_diff:.2e})")


if __name__ == "__main__":
    main()
