"""Run the frozen quality/compute protocol and write a frontier report.

    python -m self_improving_diffusion.eval.cli --out reports/jx02_quality_compute.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .quality_compute import PROTOCOL_VERSION, QualityComputeProtocol


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("reports/jx02_quality_compute.json"))
    args = parser.parse_args(argv)

    protocol = QualityComputeProtocol()
    records = protocol.run()
    frontier = protocol.frontier(records)

    report = {
        "protocol_version": PROTOCOL_VERSION,
        "records": [r.as_dict() for r in records],
        "frontier": frontier,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.out}")
    for point in frontier:
        print(f"steps={point['steps']:>3} mean_quality={point['mean_quality']:.4f} n={point['n']}")


if __name__ == "__main__":
    main()
