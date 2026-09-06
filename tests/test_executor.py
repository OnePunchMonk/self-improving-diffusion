from self_improving_diffusion import (
    DeterministicBackend,
    GenerationSpec,
    ModelRef,
    ProgramBuilder,
    ProgramExecutor,
)
from self_improving_diffusion.types import Budget, TrainingConsent


def build_program():
    return (
        ProgramBuilder(
            GenerationSpec("a red cube on a blue table", ModelRef("base", "r17"), seed=42),
            Budget(max_samples=2, max_total_steps=48),
        )
        .sample(checkpoint_at=(12,))
        .sample(checkpoint_at=(12,))
        .score(ModelRef("critic", "r6"))
        .choose(by="alignment")
        .emit_trace(TrainingConsent.OPT_IN)
        .build()
    )


def test_execution_is_deterministic_and_has_complete_provenance() -> None:
    backend = DeterministicBackend()
    executor = ProgramExecutor(backend, backend)

    first = executor.execute(build_program())
    second = executor.execute(build_program())

    assert first == second
    assert len(first.samples) == 2
    assert len(first.checkpoints) == 2
    assert len(first.scores) == 2
    assert first.estimated_denoising_steps == 48
    assert first.training_consent == "opt_in"


def test_choose_respects_the_declared_score_dimension() -> None:
    backend = DeterministicBackend()
    trace = ProgramExecutor(backend, backend).execute(build_program())

    expected = max(trace.scores, key=lambda score: score.alignment)
    assert trace.selected_sample_id == expected.sample_id
