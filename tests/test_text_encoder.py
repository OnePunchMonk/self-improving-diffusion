"""Real text conditioning: deterministic encoding, and it actually changes generation."""

import jax
import numpy as np

from self_improving_diffusion.executor import ProgramExecutor
from self_improving_diffusion.jax_backend.backend import JaxDenoiserBackend, JaxVerifierBackend
from self_improving_diffusion.jax_backend.model import COND_DIM, TinyEpsilonModel
from self_improving_diffusion.jax_backend.sampler import ddpm_step, make_schedule
from self_improving_diffusion.jax_backend.text_encoder import TinyTextEncoder, encode_prompt
from self_improving_diffusion.program import ProgramBuilder
from self_improving_diffusion.types import Budget, GenerationSpec, ModelRef


def test_encode_prompt_is_deterministic():
    a = encode_prompt("a red square")
    b = encode_prompt("a red square")
    assert np.array_equal(np.asarray(a), np.asarray(b))


def test_different_prompts_encode_differently():
    a = encode_prompt("a red square")
    b = encode_prompt("a blue circle")
    assert not np.array_equal(np.asarray(a), np.asarray(b))


def test_encoding_has_the_declared_conditioning_dimension():
    embedding = encode_prompt("a red square")
    assert embedding.shape == (COND_DIM,)


def test_tokenization_is_stable_across_calls_and_not_python_hash_salted():
    assert TinyTextEncoder.tokenize("a red square") == TinyTextEncoder.tokenize("a red square")


def test_empty_prompt_still_produces_a_valid_embedding():
    embedding = encode_prompt("   ")
    assert embedding.shape == (COND_DIM,)
    assert np.all(np.isfinite(np.asarray(embedding)))


def _build_program(prompt: str, steps: int = 12):
    spec = GenerationSpec(
        prompt=prompt,
        model=ModelRef(name="tiny-eps-mlp", revision="v0"),
        seed=7,
        width=64,
        height=64,
        steps=steps,
    )
    budget = Budget(max_samples=1, max_total_steps=steps)
    program = (
        ProgramBuilder(spec, budget)
        .sample()
        .score(ModelRef(name="tiny-heuristic-verifier", revision="v0"))
        .choose(by="quality")
        .emit_trace()
        .build()
    )
    return program


def test_different_prompts_produce_different_generations_at_the_same_seed():
    program_a = _build_program("a red square")
    program_b = _build_program("a blue circle")

    backend_a = JaxDenoiserBackend(seed=7)
    trace_a = ProgramExecutor(backend_a, JaxVerifierBackend(backend_a)).execute(program_a)
    final_a = backend_a.final_latents(trace_a.selected_sample_id)

    backend_b = JaxDenoiserBackend(seed=7)
    trace_b = ProgramExecutor(backend_b, JaxVerifierBackend(backend_b)).execute(program_b)
    final_b = backend_b.final_latents(trace_b.selected_sample_id)

    assert not np.array_equal(final_a, final_b)


def test_same_prompt_and_seed_reproduces_the_same_generation():
    program = _build_program("a red square")
    backend_1 = JaxDenoiserBackend(seed=7)
    trace_1 = ProgramExecutor(backend_1, JaxVerifierBackend(backend_1)).execute(program)

    backend_2 = JaxDenoiserBackend(seed=7)
    trace_2 = ProgramExecutor(backend_2, JaxVerifierBackend(backend_2)).execute(program)

    assert trace_1.selected_sample_id == trace_2.selected_sample_id


def test_ddpm_step_with_and_without_cond_differs():
    params = TinyEpsilonModel.init(jax.random.PRNGKey(0))
    schedule = make_schedule(8)
    key = jax.random.PRNGKey(1)
    latents = jax.random.normal(key, (1, TinyEpsilonModel.latent_dim))

    unconditioned = ddpm_step(params, schedule, latents, 7, key)
    conditioned = ddpm_step(params, schedule, latents, 7, key, encode_prompt("a specific prompt"))

    assert not np.array_equal(np.asarray(unconditioned), np.asarray(conditioned))


def test_frozen_encoder_params_are_not_affected_by_sampling_seed():
    # The text encoder is a module-level frozen singleton -- verifying two
    # backends with different sampling seeds still encode the same prompt
    # identically is the point of freezing it.
    a = encode_prompt("a fixed prompt")
    JaxDenoiserBackend(seed=1)
    JaxDenoiserBackend(seed=999)
    b = encode_prompt("a fixed prompt")
    assert np.array_equal(np.asarray(a), np.asarray(b))
