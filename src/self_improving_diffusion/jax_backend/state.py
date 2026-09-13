"""Exact serialization of real trajectory state, for real checkpoint/resume.

Unlike ``executor.Checkpoint`` (a digest over synthetic fixture data), the
state saved here is enough to *actually* continue a trajectory: the latents
array, the step it was captured at, and the identifiers needed to reconstruct
the same model/schedule/base-key. ``state_digest`` is a real sha256 over the
serialized array bytes plus metadata, not a placeholder.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TrajectoryState:
    sample_id: str
    step: int
    base_seed: int
    model_revision: str
    latents: np.ndarray

    def state_digest(self) -> str:
        hasher = sha256()
        hasher.update(self.latents.astype(np.float32).tobytes())
        hasher.update(
            json.dumps(
                {
                    "sample_id": self.sample_id,
                    "step": self.step,
                    "base_seed": self.base_seed,
                    "model_revision": self.model_revision,
                },
                sort_keys=True,
            ).encode()
        )
        return hasher.hexdigest()


def save_checkpoint(path: Path, state: TrajectoryState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path.with_suffix(".npz"), latents=state.latents)
    meta = {
        "sample_id": state.sample_id,
        "step": state.step,
        "base_seed": state.base_seed,
        "model_revision": state.model_revision,
        "state_digest": state.state_digest(),
    }
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))


def load_checkpoint(path: Path) -> TrajectoryState:
    meta = json.loads(path.with_suffix(".json").read_text())
    with np.load(path.with_suffix(".npz")) as archive:
        latents = archive["latents"]
    state = TrajectoryState(
        sample_id=meta["sample_id"],
        step=meta["step"],
        base_seed=meta["base_seed"],
        model_revision=meta["model_revision"],
        latents=latents,
    )
    if state.state_digest() != meta["state_digest"]:
        raise ValueError(f"checkpoint at {path} failed integrity check")
    return state
