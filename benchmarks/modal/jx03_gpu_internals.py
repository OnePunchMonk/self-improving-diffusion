"""JX-03: GPU internals through measured kernels, run on a cheap Modal T4.

Installs the pushed self-improving-diffusion repo, re-runs the JX-02 CPU
baseline harness on an actual GPU for a real CPU-vs-GPU comparison, then runs
two of JX-03's five exercises (elementwise fusion, tiled GEMM) with a
derived roofline (bytes/FLOPs, arithmetic intensity) and a correctness
oracle against NumPy -- the minimum evidence bar the roadmap asks for before
any kernel-level claim.

Nsight Compute (`ncu`) needs privileged perf-counter access this container
doesn't have, so kernel-level profiler evidence isn't captured here --
wall-clock + roofline arithmetic is the fallback, and that gap is reported
explicitly rather than silently skipped.
"""

import json
import subprocess
from pathlib import Path

import modal

app = modal.App("sid-jx03-gpu-internals")

REPO_ROOT = Path(__file__).resolve().parents[2]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy")
    .pip_install("jax[cuda12]")
    .add_local_dir(
        str(REPO_ROOT),
        "/root/sid",
        copy=True,
        ignore=[".git", ".venv", "__pycache__", "*.egg-info"],
    )
    .run_commands("pip install -e /root/sid")
)


@app.function(gpu="T4", image=image, timeout=600)
def run_jx03() -> dict:
    import time

    import jax
    import jax.numpy as jnp
    import numpy as np

    result: dict = {"devices": [str(d) for d in jax.devices()]}

    nvidia_smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    result["nvidia_smi"] = nvidia_smi.stdout.strip()

    # --- JX-02 baseline, re-run on GPU for a real CPU-vs-GPU comparison ---
    from self_improving_diffusion.jax_backend.benchmark import run_single_device_baseline
    from self_improving_diffusion.jax_backend.model import TinyEpsilonModel
    from self_improving_diffusion.jax_backend.sampler import make_schedule

    params = TinyEpsilonModel.init(jax.random.PRNGKey(0))
    schedule = make_schedule(24)
    jit_cache: dict = {}
    cold = run_single_device_baseline(params, schedule, seed=0, steps=24, cache_state="cold", jit_cache=jit_cache)
    warm = run_single_device_baseline(params, schedule, seed=0, steps=24, cache_state="warm", jit_cache=jit_cache)
    result["jx02_tiny_model_gpu"] = {"cold": cold.as_dict(), "warm": warm.as_dict()}

    # --- JX-03 exercise 1: elementwise fusion ---
    n = 1 << 24  # 16M float32 elements, ~64MB -- big enough to be memory-bound
    x = jax.random.normal(jax.random.PRNGKey(1), (n,), dtype=jnp.float32)

    def fused(x):
        return jnp.tanh(x * 2.0 + 1.0) * jnp.exp(-x * x)

    def unfused(x):
        a = x * 2.0
        b = a + 1.0
        c = jnp.tanh(b)
        d = x * x
        e = -d
        f = jnp.exp(e)
        return c * f

    fused_jit = jax.jit(fused)
    unfused_jit = jax.jit(unfused)
    jax.block_until_ready(fused_jit(x))
    jax.block_until_ready(unfused_jit(x))

    def timed(fn, x, reps=20):
        start = time.perf_counter()
        for _ in range(reps):
            out = fn(x)
        jax.block_until_ready(out)
        return (time.perf_counter() - start) / reps

    fused_s = timed(fused_jit, x)
    unfused_s = timed(unfused_jit, x)
    bytes_moved = n * 4 * 2  # one read + one write, fp32
    result["elementwise_fusion"] = {
        "n_elements": n,
        "fused_s": fused_s,
        "unfused_s": unfused_s,
        "speedup": unfused_s / fused_s,
        "fused_achieved_gbps": bytes_moved / fused_s / 1e9,
        "unfused_achieved_gbps": bytes_moved / unfused_s / 1e9,
        "note": "unfused chain still gets fused by XLA's compiler in practice; "
        "a real win here needs deliberately compiler-opaque boundaries (e.g. donated buffers or "
        "explicit host round-trips), which this measurement does not yet force.",
    }

    # --- JX-03 exercise 2: tiled GEMM, correctness + roofline ---
    for size, dtype_name in [(1024, "float32"), (4096, "float32"), (4096, "bfloat16")]:
        dtype = jnp.float32 if dtype_name == "float32" else jnp.bfloat16
        a = jax.random.normal(jax.random.PRNGKey(2), (size, size), dtype=dtype)
        b = jax.random.normal(jax.random.PRNGKey(3), (size, size), dtype=dtype)
        matmul = jax.jit(lambda a, b: a @ b)
        jax.block_until_ready(matmul(a, b))

        reps = 10
        start = time.perf_counter()
        for _ in range(reps):
            out = matmul(a, b)
        jax.block_until_ready(out)
        elapsed_s = (time.perf_counter() - start) / reps

        flops = 2 * size**3
        achieved_tflops = flops / elapsed_s / 1e12

        # correctness oracle: NumPy fp32 reference at the same size (cast up for bf16)
        a_np = np.asarray(a, dtype=np.float32)
        b_np = np.asarray(b, dtype=np.float32)
        ref = a_np @ b_np
        got = np.asarray(out, dtype=np.float32)
        rel_err = float(np.max(np.abs(got - ref)) / (np.max(np.abs(ref)) + 1e-8))

        result.setdefault("tiled_gemm", []).append(
            {
                "size": size,
                "dtype": dtype_name,
                "elapsed_s": elapsed_s,
                "achieved_tflops": achieved_tflops,
                "max_rel_err_vs_numpy_fp32": rel_err,
                "arithmetic_intensity_flops_per_byte": flops
                / (3 * size**2 * (4 if dtype_name == "float32" else 2)),
            }
        )

    result["ncu_profiler_evidence"] = "not captured: this container lacks the privileged " \
        "perf-counter access Nsight Compute needs; wall-clock + roofline arithmetic used instead."

    return result


@app.local_entrypoint()
def main():
    result = run_jx03.remote()
    print(json.dumps(result, indent=2))
