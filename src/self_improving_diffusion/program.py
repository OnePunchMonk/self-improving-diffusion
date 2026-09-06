"""A deliberately small, typed DSL for bounded generation programs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .types import Budget, GenerationSpec, ModelRef, TrainingConsent


class NodeKind(str, Enum):
    SAMPLE = "sample"
    SCORE = "score"
    CHOOSE = "choose"
    EMIT_TRACE = "emit_trace"


@dataclass(frozen=True)
class ProgramNode:
    """An immutable operation in a generation program."""

    kind: NodeKind
    config: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "config": self.config}


@dataclass(frozen=True)
class GenerationProgram:
    """A compiled program with all model references and cost limits pinned."""

    spec: GenerationSpec
    budget: Budget
    nodes: tuple[ProgramNode, ...]
    consent: TrainingConsent

    def as_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.as_dict(),
            "budget": self.budget.as_dict(),
            "nodes": [node.as_dict() for node in self.nodes],
            "consent": self.consent.value,
        }


class ProgramBuilder:
    """Build the v0 `sample -> score -> choose -> emit_trace` program.

    The builder rejects implicit or unbounded search: callers must declare the
    number of samples and the total denoising-step budget before compiling.
    """

    def __init__(self, spec: GenerationSpec, budget: Budget | None = None) -> None:
        self._spec = spec
        self._budget = budget or Budget(max_total_steps=spec.steps)
        self._nodes: list[ProgramNode] = []
        self._consent = TrainingConsent.DISABLED
        self._sample_count = 0

    def sample(self, *, checkpoint_at: tuple[int, ...] = ()) -> "ProgramBuilder":
        if self._sample_count >= self._budget.max_samples:
            raise ValueError("sample count would exceed max_samples")
        if any(step < 1 or step >= self._spec.steps for step in checkpoint_at):
            raise ValueError("checkpoints must be inside the denoising trajectory")
        self._sample_count += 1
        self._nodes.append(
            ProgramNode(
                NodeKind.SAMPLE,
                {"checkpoint_at": list(checkpoint_at), "sample_index": self._sample_count},
            )
        )
        return self

    def score(self, verifier: ModelRef) -> "ProgramBuilder":
        self._nodes.append(ProgramNode(NodeKind.SCORE, {"verifier": verifier.identifier}))
        return self

    def choose(self, *, by: str = "quality") -> "ProgramBuilder":
        if by not in {"quality", "alignment"}:
            raise ValueError("choose criterion must be quality or alignment")
        self._nodes.append(ProgramNode(NodeKind.CHOOSE, {"by": by}))
        return self

    def emit_trace(
        self, consent: TrainingConsent = TrainingConsent.DISABLED
    ) -> "ProgramBuilder":
        self._consent = consent
        self._nodes.append(ProgramNode(NodeKind.EMIT_TRACE, {"consent": consent.value}))
        return self

    def build(self) -> GenerationProgram:
        if not self._nodes or self._nodes[0].kind is not NodeKind.SAMPLE:
            raise ValueError("a program must begin with sample")
        if self._sample_count == 0:
            raise ValueError("a program requires at least one sample")
        if self._sample_count * self._spec.steps > self._budget.max_total_steps:
            raise ValueError("program would exceed max_total_steps")
        if not any(node.kind is NodeKind.EMIT_TRACE for node in self._nodes):
            raise ValueError("a program must explicitly emit a trace")
        self._validate_execution_order()
        return GenerationProgram(
            spec=self._spec,
            budget=self._budget,
            nodes=tuple(self._nodes),
            consent=self._consent,
        )

    def _validate_execution_order(self) -> None:
        """Reject valid-looking programs that would be semantically ambiguous."""

        saw_sample = False
        saw_score = False
        saw_choose = False
        for index, node in enumerate(self._nodes):
            if node.kind is NodeKind.SAMPLE:
                if saw_score:
                    raise ValueError("v0 requires all samples before score")
                saw_sample = True
            elif node.kind is NodeKind.SCORE:
                if not saw_sample:
                    raise ValueError("score requires a preceding sample")
                if saw_score:
                    raise ValueError("v0 permits one scoring stage")
                saw_score = True
            elif node.kind is NodeKind.CHOOSE:
                if not saw_score:
                    raise ValueError("choose requires a preceding score")
                if saw_choose:
                    raise ValueError("v0 permits one selection stage")
                saw_choose = True
            elif node.kind is NodeKind.EMIT_TRACE and index != len(self._nodes) - 1:
                raise ValueError("emit_trace must terminate a program")
