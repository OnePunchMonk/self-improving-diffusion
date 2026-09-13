"""Run JX-07's bounded self-improvement loop and write evidence.

    python -m self_improving_diffusion.rsi.cli --cycles 2 --out reports/jx07_self_improvement.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .controller import run_multiple_cycles


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("reports/jx07_self_improvement.json"))
    args = parser.parse_args(argv)

    report = run_multiple_cycles(num_cycles=args.cycles)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report.as_dict(), indent=2))

    print(f"wrote {args.out}")
    for cycle in report.cycles:
        print(
            f"cycle {cycle.cycle_index}: candidate n={cycle.candidate_n} "
            f"(vs baseline n={cycle.baseline_n}) promoted={cycle.promoted} "
            f"candidate_q={cycle.candidate_selection_quality:.4f} baseline_q={cycle.baseline_selection_quality:.4f}"
        )
    print(
        f"final active n={report.final_active_n} quality={report.final_active_quality:.4f} "
        f"vs frozen baseline (n={3}) quality={report.final_frozen_baseline_quality:.4f}"
    )
    print(f"sustained improvement: {report.final_active_quality > report.final_frozen_baseline_quality}")
    print(f"negative transfer detected: {report.negative_transfer_detected}")
    print(f"total offline cost: {report.total_offline_cost_steps} steps")


if __name__ == "__main__":
    main()
