"""JX-01: the JAX backend must satisfy the same protocol the executor already tests against."""

import numpy as np

from self_improving_diffusion.executor import ProgramExecutor
from self_improving_diffusion.jax_backend.backend import JaxDenoiserBackend, JaxVerifierBackend
from self_improving_diffusion.program import ProgramBuilder
from self_improving_diffusion.types import Budget, GenerationSpec, ModelRef


def _build_program(steps: int = 8):
    spec = GenerationSpec(
        prompt="unit test prompt",
        model=ModelRef(name="tiny-eps-mlp", revision="v0"),
        seed=42,
        width=64,
        height=64,
        steps=steps,
    )
    budget = Budget(max_samples=1, max_total_steps=steps)
    program = (
        ProgramBuilder(spec, budget)
        .sample(checkpoint_at=(steps // 2,))
        .score(ModelRef(name="tiny-heuristic-verifier", revision="v0"))
        .choose(by="quality")
        .emit_trace()
        .build()
    )
    return spec, program


def test_executor_runs_against_real_jax_backend():
    spec, program = _build_program()
    backend = JaxDenoiserBackend(seed=spec.seed)
    verifier = JaxVerifierBackend(backend)
    executor = ProgramExecutor(backend, verifier)

    trace = executor.execute(program)

    assert len(trace.samples) == 1
    assert len(trace.checkpoints) == 1
    assert trace.checkpoints[0].step == program.spec.steps // 2
    final = backend.final_latents(trace.selected_sample_id)
    assert final.shape == (1, 64)
    assert np.all(np.isfinite(final))


def test_same_seed_and_program_produce_same_selected_sample_id():
    spec, program = _build_program()
    backend_a = JaxDenoiserBackend(seed=spec.seed)
    backend_b = JaxDenoiserBackend(seed=spec.seed)
    trace_a = ProgramExecutor(backend_a, JaxVerifierBackend(backend_a)).execute(program)
    trace_b = ProgramExecutor(backend_b, JaxVerifierBackend(backend_b)).execute(program)
    assert trace_a.selected_sample_id == trace_b.selected_sample_id


def test_checkpoint_state_digest_reflects_real_latents():
    spec, program = _build_program()
    backend = JaxDenoiserBackend(seed=spec.seed)
    trace = ProgramExecutor(backend, JaxVerifierBackend(backend)).execute(program)
    checkpoint = trace.checkpoints[0]

    other_checkpoint = backend.checkpoint(trace.samples[0], step=checkpoint.step + 1)
    assert other_checkpoint.state_digest != checkpoint.state_digest
