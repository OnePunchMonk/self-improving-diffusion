"""Run JX-06's four budgeted-inference policies on the held-out seed set.

    python -m self_improving_diffusion.eval.adaptive_cli --out reports/jx06_adaptive_inference.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .adaptive_inference import ADAPTIVE_ACCEPT_THRESHOLD, PROBE_STEPS, TOTAL_STEP_BUDGET, run_all_policies, summarize


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("reports/jx06_adaptive_inference.json"))
    args = parser.parse_args(argv)

    outcomes = run_all_policies()
    summary = summarize(outcomes)

    report = {
        "total_step_budget": TOTAL_STEP_BUDGET,
        "probe_steps": PROBE_STEPS,
        "adaptive_accept_threshold": ADAPTIVE_ACCEPT_THRESHOLD,
        "records": [asdict(o) for o in outcomes],
        "summary": summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    for row in sorted(summary, key=lambda r: -r["mean_quality"]):
        print(
            f"{row['policy']:>14}: mean_quality={row['mean_quality']:.4f} "
            f"mean_cost_steps={row['mean_total_cost_steps']:.1f} "
            f"mean_samples={row['mean_samples_drawn']:.2f}"
        )


if __name__ == "__main__":
    main()
