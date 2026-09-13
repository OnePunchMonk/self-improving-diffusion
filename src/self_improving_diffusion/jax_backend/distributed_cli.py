"""JX-04 evidence command: distributed correctness + collective latency sweep.

Requires >= 2 devices, simulated via:

    XLA_FLAGS=--xla_force_host_platform_device_count=4 \
        python -m self_improving_diffusion.jax_backend.distributed_cli --out reports/jx04_distributed.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import jax

from .distributed import measure_collective_latency, run_data_parallel_samples
from .model import TinyEpsilonModel
from .sampler import make_schedule


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--out", type=Path, default=Path("reports/jx04_distributed.json"))
    args = parser.parse_args(argv)

    num_devices = len(jax.local_devices())
    if num_devices < 2:
        raise SystemExit(
            "need >= 2 devices; set XLA_FLAGS=--xla_force_host_platform_device_count=N "
            "before running (N >= 2), since JAX can't add devices after init"
        )

    params = TinyEpsilonModel.init(jax.random.PRNGKey(args.seed))
    schedule = make_schedule(args.steps)

    _, report = run_data_parallel_samples(
        params, schedule, seed=args.seed, request_id="jx04-evidence", num_samples=args.num_samples
    )
    collective = measure_collective_latency()

    output = {
        "num_devices_available": num_devices,
        "data_parallel_run": asdict(report),
        "collective_latency_by_message_size": collective,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2))
    print(f"wrote {args.out}")
    print(f"devices: {num_devices}, matches single-device ground truth: {report.matches_single_device}")


if __name__ == "__main__":
    main()
