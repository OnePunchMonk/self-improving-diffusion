import json

import pytest

from self_improving_diffusion import (
    DeterministicBackend,
    GenerationSpec,
    JsonlEventWriter,
    ModelRef,
    ProgramBuilder,
    ProgramExecutor,
    TrainingConsent,
    event_from_trace,
)


def executed_event(consent: TrainingConsent):
    program = (
        ProgramBuilder(GenerationSpec("confidential prompt", ModelRef("base", "r17"), 9))
        .sample()
        .score(ModelRef("critic", "r6"))
        .choose()
        .emit_trace(consent)
        .build()
    )
    backend = DeterministicBackend()
    trace = ProgramExecutor(backend, backend).execute(program)
    return event_from_trace(program, trace)


def test_opted_in_event_is_data_minimised_and_writable(tmp_path) -> None:
    event = executed_event(TrainingConsent.OPT_IN)
    path = tmp_path / "events" / "learning.jsonl"

    JsonlEventWriter(path).write(event)

    record = json.loads(path.read_text())
    assert record["training_eligible"] is True
    assert "confidential prompt" not in path.read_text()
    assert record["model_revision"] == "base@r17"


def test_event_writer_rejects_non_opted_in_trace(tmp_path) -> None:
    event = executed_event(TrainingConsent.DISABLED)

    with pytest.raises(PermissionError, match="explicit opt-in"):
        JsonlEventWriter(tmp_path / "events.jsonl").write(event)
