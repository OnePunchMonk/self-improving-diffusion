"""The frozen quality-versus-compute evaluation protocol referenced by JX-02.

JX-06's acceptance criterion is "held-out quality/compute frontier beats the
simple baseline, or publish why it does not" -- but that comparison is only
meaningful if the protocol producing it is declared once and not
re-tuned per experiment. This module is that declaration: a fixed set of
seeds and step budgets, one verifier revision, one selection criterion.
Changing any of those is a new protocol version, not a silent edit.

v0 only has the tiny JX-01 model and a heuristic (non-learned) verifier, so
this cannot yet make a real quality claim -- it exists so the harness and
report shape are already correct before a real verifier is swapped in.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import time

import jax

from ..executor import ProgramExecutor
from ..jax_backend.backend import JaxDenoiserBackend, JaxVerifierBackend
from ..program import ProgramBuilder
from ..types import Budget, GenerationSpec, ModelRef

PROTOCOL_VERSION = "quality-compute-v0"

# Frozen at protocol authoring time. Do not tune these to make a result look
# better; bump PROTOCOL_VERSION and note why if they ever need to change.
FROZEN_SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5)
FROZEN_STEP_BUDGETS: tuple[int, ...] = (4, 8, 16, 32)
FROZEN_MODEL = ModelRef(name="tiny-eps-mlp", revision="v0")
FROZEN_VERIFIER = ModelRef(name="tiny-heuristic-verifier", revision="v0")


@dataclass(frozen=True)
class QualityComputeRecord:
    protocol_version: str
    seed: int
    steps: int
    quality: float
    alignment: float
    uncertainty: float
    estimated_denoising_steps: int
    wall_time_s: float

    def as_dict(self) -> dict:
        return asdict(self)


class QualityComputeProtocol:
    """Run the frozen seed x step-budget grid and report the resulting frontier."""

    def __init__(
        self,
        seeds: tuple[int, ...] = FROZEN_SEEDS,
        step_budgets: tuple[int, ...] = FROZEN_STEP_BUDGETS,
    ) -> None:
        self._seeds = seeds
        self._step_budgets = step_budgets

    def run(self) -> list[QualityComputeRecord]:
        records: list[QualityComputeRecord] = []
        for steps in self._step_budgets:
            for seed in self._seeds:
                records.append(self._run_one(seed=seed, steps=steps))
        return records

    def _run_one(self, seed: int, steps: int) -> QualityComputeRecord:
        spec = GenerationSpec(
            prompt="frozen quality/compute protocol probe",
            model=FROZEN_MODEL,
            seed=seed,
            width=64,
            height=64,
            steps=steps,
        )
        budget = Budget(max_samples=1, max_total_steps=steps)
        program = (
            ProgramBuilder(spec, budget)
            .sample()
            .score(FROZEN_VERIFIER)
            .choose(by="quality")
            .emit_trace()
            .build()
        )
        backend = JaxDenoiserBackend(seed=seed)
        verifier = JaxVerifierBackend(backend)
        executor = ProgramExecutor(backend, verifier)

        start = time.perf_counter()
        trace = executor.execute(program)
        jax.block_until_ready(backend.final_latents(trace.selected_sample_id))
        wall_time_s = time.perf_counter() - start

        score = trace.scores[0]
        return QualityComputeRecord(
            protocol_version=PROTOCOL_VERSION,
            seed=seed,
            steps=steps,
            quality=score.quality,
            alignment=score.alignment,
            uncertainty=score.uncertainty,
            estimated_denoising_steps=trace.estimated_denoising_steps,
            wall_time_s=wall_time_s,
        )

    @staticmethod
    def frontier(records: list[QualityComputeRecord]) -> list[dict]:
        """Mean quality per step budget, sorted by compute (ascending)."""

        by_steps: dict[int, list[QualityComputeRecord]] = {}
        for record in records:
            by_steps.setdefault(record.steps, []).append(record)

        frontier = []
        for steps in sorted(by_steps):
            group = by_steps[steps]
            frontier.append(
                {
                    "steps": steps,
                    "mean_quality": sum(r.quality for r in group) / len(group),
                    "mean_wall_time_s": sum(r.wall_time_s for r in group) / len(group),
                    "n": len(group),
                }
            )
        return frontier
