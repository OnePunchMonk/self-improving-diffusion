import pytest

from self_improving_diffusion.program import NodeKind, ProgramBuilder
from self_improving_diffusion.types import Budget, GenerationSpec, ModelRef, TrainingConsent


def spec() -> GenerationSpec:
    return GenerationSpec(
        prompt="a red cube on a blue table",
        model=ModelRef("base", "r17"),
        seed=42,
    )


def test_program_pins_versions_and_serializes_trace_consent() -> None:
    program = (
        ProgramBuilder(spec())
        .sample(checkpoint_at=(12,))
        .score(ModelRef("critic", "r6"))
        .choose()
        .emit_trace(TrainingConsent.OPT_IN)
        .build()
    )

    assert [node.kind for node in program.nodes] == [
        NodeKind.SAMPLE,
        NodeKind.SCORE,
        NodeKind.CHOOSE,
        NodeKind.EMIT_TRACE,
    ]
    assert program.as_dict()["consent"] == "opt_in"
    assert program.spec.compatibility_key()[0] == "base@r17"


def test_program_rejects_unbounded_sample_expansion() -> None:
    with pytest.raises(ValueError, match="max_samples"):
        ProgramBuilder(spec(), Budget(max_samples=1, max_total_steps=48)).sample().sample()


def test_program_rejects_total_step_budget_exhaustion() -> None:
    with pytest.raises(ValueError, match="max_total_steps"):
        (
            ProgramBuilder(spec(), Budget(max_samples=2, max_total_steps=24))
            .sample()
            .sample()
            .emit_trace()
            .build()
        )


def test_generation_spec_requires_latent_aligned_dimensions() -> None:
    with pytest.raises(ValueError, match="divisible by 8"):
        GenerationSpec("test", ModelRef("base", "r17"), 1, width=1025)
