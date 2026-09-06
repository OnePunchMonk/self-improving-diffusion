"""Deterministic local executor for the first program vertical slice.

This module is intentionally model-agnostic.  It establishes the execution,
checkpoint, selection, and trace contracts that a GPU denoiser backend will
later implement.  Its deterministic backend is useful for replay tests and for
validating the control plane without a model download.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Protocol

from .program import GenerationProgram, NodeKind


@dataclass(frozen=True)
class SampleArtifact:
    """A model result represented by a stable content address in v0."""

    sample_id: str
    sample_index: int
    model_revision: str
    steps: int


@dataclass(frozen=True)
class Checkpoint:
    """Resume metadata for one exact trajectory boundary."""

    sample_id: str
    step: int
    state_digest: str


@dataclass(frozen=True)
class ScoreReport:
    sample_id: str
    verifier_revision: str
    quality: float
    alignment: float
    uncertainty: float


@dataclass(frozen=True)
class ExecutionTrace:
    """A complete, serializable record of one program execution."""

    program_digest: str
    selected_sample_id: str
    samples: tuple[SampleArtifact, ...]
    checkpoints: tuple[Checkpoint, ...]
    scores: tuple[ScoreReport, ...]
    estimated_denoising_steps: int
    training_consent: str


class DenoiserBackend(Protocol):
    def sample(self, program: GenerationProgram, sample_index: int) -> SampleArtifact:
        ...

    def checkpoint(self, sample: SampleArtifact, step: int) -> Checkpoint:
        ...


class VerifierBackend(Protocol):
    def score(self, sample: SampleArtifact, verifier_revision: str) -> ScoreReport:
        ...


class DeterministicBackend:
    """Reference backend: stable pseudo-artifacts, never image generation."""

    def sample(self, program: GenerationProgram, sample_index: int) -> SampleArtifact:
        payload = {
            "spec": program.spec.as_dict(),
            "sample_index": sample_index,
            "program": program.as_dict(),
        }
        digest = _digest(payload)
        return SampleArtifact(
            sample_id=f"sample_{digest[:20]}",
            sample_index=sample_index,
            model_revision=program.spec.model.identifier,
            steps=program.spec.steps,
        )

    def checkpoint(self, sample: SampleArtifact, step: int) -> Checkpoint:
        return Checkpoint(
            sample_id=sample.sample_id,
            step=step,
            state_digest=_digest({"sample_id": sample.sample_id, "step": step}),
        )

    def score(self, sample: SampleArtifact, verifier_revision: str) -> ScoreReport:
        digest = sha256(f"{sample.sample_id}:{verifier_revision}".encode()).digest()
        quality = int.from_bytes(digest[:4], "big") / 2**32
        alignment = int.from_bytes(digest[4:8], "big") / 2**32
        uncertainty = int.from_bytes(digest[8:12], "big") / 2**32
        return ScoreReport(
            sample_id=sample.sample_id,
            verifier_revision=verifier_revision,
            quality=quality,
            alignment=alignment,
            uncertainty=uncertainty,
        )


class ProgramExecutor:
    """Execute a bounded program and return a complete replayable trace."""

    def __init__(self, denoiser: DenoiserBackend, verifier: VerifierBackend) -> None:
        self._denoiser = denoiser
        self._verifier = verifier

    def execute(self, program: GenerationProgram) -> ExecutionTrace:
        samples: list[SampleArtifact] = []
        checkpoints: list[Checkpoint] = []
        scores: list[ScoreReport] = []
        selected: SampleArtifact | None = None
        choice = "quality"

        for node in program.nodes:
            if node.kind is NodeKind.SAMPLE:
                sample = self._denoiser.sample(program, node.config["sample_index"])
                samples.append(sample)
                for step in node.config["checkpoint_at"]:
                    checkpoints.append(self._denoiser.checkpoint(sample, step))
            elif node.kind is NodeKind.SCORE:
                verifier_revision = node.config["verifier"]
                scores = [self._verifier.score(sample, verifier_revision) for sample in samples]
            elif node.kind is NodeKind.CHOOSE:
                if not scores:
                    raise ValueError("choose requires at least one score report")
                choice = node.config["by"]
                score_by_id = {score.sample_id: score for score in scores}
                selected = max(samples, key=lambda item: getattr(score_by_id[item.sample_id], choice))
            elif node.kind is NodeKind.EMIT_TRACE:
                continue
            else:  # pragma: no cover - protects future NodeKind extensions
                raise ValueError(f"unsupported node kind: {node.kind}")

        if selected is None:
            selected = samples[-1]
        return ExecutionTrace(
            program_digest=_digest(program.as_dict()),
            selected_sample_id=selected.sample_id,
            samples=tuple(samples),
            checkpoints=tuple(checkpoints),
            scores=tuple(scores),
            estimated_denoising_steps=sum(sample.steps for sample in samples),
            training_consent=program.consent.value,
        )


def _digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return sha256(encoded).hexdigest()
