"""Stable value types shared across the API, engine, and event log.

These types intentionally use only the Python standard library.  The first
vertical slice should be easy to replay without pulling a model runtime into
the control-plane package.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any


class TrainingConsent(str, Enum):
    """Whether a generation trace may enter the learning-data pipeline."""

    DISABLED = "disabled"
    OPT_IN = "opt_in"


@dataclass(frozen=True)
class ModelRef:
    """An immutable reference to a model or verifier artifact."""

    name: str
    revision: str

    def __post_init__(self) -> None:
        if not self.name or not self.revision:
            raise ValueError("model name and revision must both be non-empty")

    @property
    def identifier(self) -> str:
        return f"{self.name}@{self.revision}"


@dataclass(frozen=True)
class GenerationSpec:
    """All information needed to reproduce a denoising trajectory."""

    prompt: str
    model: ModelRef
    seed: int
    width: int = 1024
    height: int = 1024
    steps: int = 24
    scheduler: str = "ddim"
    guidance_scale: float = 7.5

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("prompt must be non-empty")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.width % 8 or self.height % 8:
            raise ValueError("width and height must be divisible by 8")
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.guidance_scale < 0:
            raise ValueError("guidance_scale must be non-negative")

    def compatibility_key(self) -> tuple[str, str, int, int, str, int, float]:
        """Return the v0 batch key; prompts and seeds never participate."""

        return (
            self.model.identifier,
            self.scheduler,
            self.width,
            self.height,
            "fp16",
            self.steps,
            self.guidance_scale,
        )

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["model"] = asdict(self.model)
        return result


@dataclass(frozen=True)
class Budget:
    """A compile-time cap that prevents an execution program from expanding."""

    max_samples: int = 1
    max_total_steps: int = 24

    def __post_init__(self) -> None:
        if self.max_samples < 1 or self.max_total_steps < 1:
            raise ValueError("budget limits must be positive")

    def as_dict(self) -> dict[str, int]:
        return asdict(self)
