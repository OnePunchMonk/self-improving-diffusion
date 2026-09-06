"""Consent-aware, append-only execution events.

Events intentionally contain provenance and aggregate scores, not raw prompts,
images, or trajectory tensors.  A future data-retention service can join those
artifacts under a separate access policy; the default learning boundary cannot.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

from .executor import ExecutionTrace
from .program import GenerationProgram
from .types import TrainingConsent


@dataclass(frozen=True)
class LearningEvent:
    """The minimal immutable record eligible for downstream pair building."""

    event_id: str
    trace_digest: str
    prompt_digest: str
    model_revision: str
    verifier_revisions: tuple[str, ...]
    selected_sample_id: str
    score_summary: tuple[dict[str, float | str], ...]
    estimated_denoising_steps: int
    training_consent: str

    @property
    def training_eligible(self) -> bool:
        return self.training_consent == TrainingConsent.OPT_IN.value

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["verifier_revisions"] = list(self.verifier_revisions)
        result["score_summary"] = list(self.score_summary)
        result["training_eligible"] = self.training_eligible
        return result


def event_from_trace(program: GenerationProgram, trace: ExecutionTrace) -> LearningEvent:
    """Create a data-minimised event after checking program/trace provenance."""

    if trace.training_consent != program.consent.value:
        raise ValueError("trace consent does not match its generation program")
    if trace.selected_sample_id not in {sample.sample_id for sample in trace.samples}:
        raise ValueError("trace selected sample is absent from its sample set")

    score_summary = tuple(
        {
            "sample_id": score.sample_id,
            "verifier_revision": score.verifier_revision,
            "quality": score.quality,
            "alignment": score.alignment,
            "uncertainty": score.uncertainty,
        }
        for score in trace.scores
    )
    trace_digest = _digest(
        {
            "program_digest": trace.program_digest,
            "selected_sample_id": trace.selected_sample_id,
            "samples": [asdict(sample) for sample in trace.samples],
            "scores": list(score_summary),
        }
    )
    return LearningEvent(
        event_id=f"event_{trace_digest[:20]}",
        trace_digest=trace_digest,
        prompt_digest=sha256(program.spec.prompt.encode()).hexdigest(),
        model_revision=program.spec.model.identifier,
        verifier_revisions=tuple(sorted({score.verifier_revision for score in trace.scores})),
        selected_sample_id=trace.selected_sample_id,
        score_summary=score_summary,
        estimated_denoising_steps=trace.estimated_denoising_steps,
        training_consent=trace.training_consent,
    )


class JsonlEventWriter:
    """Local development sink that refuses non-consented learning records."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def write(self, event: LearningEvent) -> None:
        if not event.training_eligible:
            raise PermissionError("training event requires explicit opt-in consent")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event.as_dict(), sort_keys=True, separators=(",", ":")))
            stream.write("\n")


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()
