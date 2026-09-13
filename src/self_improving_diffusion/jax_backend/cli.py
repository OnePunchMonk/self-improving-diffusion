"""One command: run a tiny real JAX generation, verify exact resume, write evidence.

    python -m self_improving_diffusion.jax_backend.cli --prompt "a red square" \
        --seed 0 --steps 24 --out-dir runs/jx01

Writes ``image.png`` (the decoded latent, nearest-neighbour upscaled),
``manifest.json`` (spec, execution trace, timings, resume-equivalence result)
into ``--out-dir``.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import jax
import numpy as np
from PIL import Image

from ..executor import ProgramExecutor
from ..program import ProgramBuilder
from ..types import Budget, GenerationSpec, ModelRef
from .backend import JaxDenoiserBackend, JaxVerifierBackend
from .sampler import make_schedule, run_trajectory


def _decode_to_image(latents: np.ndarray, scale: int = 16) -> Image.Image:
    side = int(round(latents.size**0.5))
    grid = latents.reshape(side, side)
    normalized = (grid - grid.min()) / (np.ptp(grid) + 1e-8)
    pixels = (normalized * 255).astype(np.uint8)
    image = Image.fromarray(pixels, mode="L")
    return image.resize((side * scale, side * scale), Image.NEAREST)


def _verify_exact_resume(backend: JaxDenoiserBackend, spec: GenerationSpec, sample_index: int, resume_step: int) -> bool:
    """Prove checkpoint/resume equals an uninterrupted run, bit-for-bit."""

    schedule = make_schedule(spec.steps)
    base_key = jax.random.fold_in(jax.random.PRNGKey(spec.seed), sample_index)
    init_latents = jax.random.normal(base_key, (1, backend.params().w1.shape[0] - 1))

    full_run = run_trajectory(backend.params(), schedule, init_latents, base_key, spec.steps - 1, -1)

    to_checkpoint = run_trajectory(backend.params(), schedule, init_latents, base_key, spec.steps - 1, resume_step)
    resumed = run_trajectory(backend.params(), schedule, to_checkpoint, base_key, resume_step, -1)

    return bool(np.array_equal(np.asarray(full_run), np.asarray(resumed)))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="a tiny real jax generation")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--model-revision", default="tiny-eps-mlp@v0")
    parser.add_argument("--out-dir", type=Path, default=Path("runs/jx01"))
    args = parser.parse_args(argv)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    spec = GenerationSpec(
        prompt=args.prompt,
        model=ModelRef(name="tiny-eps-mlp", revision="v0"),
        seed=args.seed,
        width=64,
        height=64,
        steps=args.steps,
        scheduler="ddpm",
    )
    budget = Budget(max_samples=1, max_total_steps=spec.steps)
    program = (
        ProgramBuilder(spec, budget)
        .sample(checkpoint_at=(spec.steps // 2,))
        .score(ModelRef(name="tiny-heuristic-verifier", revision="v0"))
        .choose(by="quality")
        .emit_trace()
        .build()
    )

    backend = JaxDenoiserBackend(seed=args.seed, checkpoint_dir=out_dir / "checkpoints")
    verifier = JaxVerifierBackend(backend)
    executor = ProgramExecutor(backend, verifier)

    compile_start = time.perf_counter()
    trace = executor.execute(program)
    jax.block_until_ready(backend.final_latents(trace.selected_sample_id))
    compile_and_run_s = time.perf_counter() - compile_start

    warm_start = time.perf_counter()
    executor.execute(program)
    warm_run_s = time.perf_counter() - warm_start

    resume_ok = _verify_exact_resume(backend, spec, sample_index=1, resume_step=spec.steps // 2)

    final_latents = backend.final_latents(trace.selected_sample_id)
    image = _decode_to_image(final_latents)
    image.save(out_dir / "image.png")

    manifest = {
        "spec": spec.as_dict(),
        "budget": budget.as_dict(),
        "trace": {
            "program_digest": trace.program_digest,
            "selected_sample_id": trace.selected_sample_id,
            "samples": [asdict(s) for s in trace.samples],
            "checkpoints": [asdict(c) for c in trace.checkpoints],
            "scores": [asdict(s) for s in trace.scores],
            "estimated_denoising_steps": trace.estimated_denoising_steps,
        },
        "timings_seconds": {
            "first_call_compile_plus_run": compile_and_run_s,
            "second_call_warm_run": warm_run_s,
        },
        "exact_resume_equivalence": resume_ok,
        "devices": [str(d) for d in jax.devices()],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"wrote {out_dir / 'image.png'} and {out_dir / 'manifest.json'}")
    print(f"exact resume equivalence: {resume_ok}")
    if not resume_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
