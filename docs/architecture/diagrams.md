# Architecture diagrams

Three views of what's actually implemented (JX-01 through JX-07), not an
aspirational sketch of the full charter. Each diagram cites the module it
describes so it can be checked against real code, not just read as prose.

## 1. Inference system (JX-01, JX-02, JX-04, JX-05)

One request's path from program compilation through distributed/serving
execution to evidence output.

```mermaid
flowchart TD
    subgraph Client["Caller"]
        REQ["GenerationSpec + Budget\n(prompt, seed, steps, model ref)"]
    end

    subgraph DSL["Program DSL — program.py"]
        PB["ProgramBuilder\n.sample() .score() .choose() .emit_trace()"]
        PROG["GenerationProgram\n(frozen: spec, budget, nodes, consent)"]
    end

    subgraph Admission["Admission & scheduling — scheduler.py, jax_backend/serving.py"]
        SBS["StepBatchScheduler\ncompatibility-key FIFO batching"]
        CVC["CompiledVariantCache\nbounded LRU, raises rather than\nunbounded recompilation"]
        POLICY{"static_batching\nvs\nchunk_scheduling"}
    end

    subgraph Exec["Execution — executor.py, jax_backend/backend.py"]
        PE["ProgramExecutor.execute()"]
        DB["JaxDenoiserBackend\n(TinyEpsilonModel + DDPM sampler)"]
        VB["JaxVerifierBackend\n(heuristic quality/alignment/uncertainty)"]
    end

    subgraph Dist["Distributed layer — jax_backend/distributed.py"]
        PMAP["jax.pmap over local devices\nkey = fold_in(seed, request_id, sample_index)\n— never rank or placement"]
    end

    subgraph State["State — jax_backend/state.py, checkpoints.py"]
        CKPT["TrajectoryState\nreal latents .npz + sha256 state_digest"]
        STORE["ExactCheckpointStore\nrequest-scoped, TTL-bounded"]
    end

    subgraph Evidence["Evidence — reports/, docs/learning/"]
        MANIFEST["manifest.json\ntrace + timings (compile/dispatch/\nexecute/transfer/decode/verify)"]
        IMG["image.png"]
    end

    REQ --> PB --> PROG --> SBS
    SBS --> CVC
    CVC --> POLICY
    POLICY --> PE
    PE --> DB
    PE --> VB
    DB --> PMAP
    DB --> CKPT --> STORE
    PE --> MANIFEST
    DB --> IMG

    classDef real fill:#e8f4ea,stroke:#2f7a3d;
    class DSL,Admission,Exec,Dist,State,Evidence real;
```

**What this leaves out on purpose**: no intra-request tensor/sequence
sharding (JX-04 explicitly defers it -- the model is too small to warrant
it), no Pallas/vendor kernel dispatch layer (JX-03 measured that XLA's
default fusion already covers the one exercise tried), no real multi-host
topology (everything above the `Dist` box has only been run on a single
host, CPU or one T4).

## 2. Self-improving loop (JX-06, JX-07)

How a serving decision (which best-of-N split to use) gets proposed,
gated, and possibly rolled back -- without ever touching model weights.

```mermaid
flowchart TD
    subgraph Frozen["Frozen for this milestone"]
        MODEL["Model parameters\n(TinyEpsilonModel weights)"]
        EXECPOL["Execution policy\n(JX-05 scheduling)"]
    end

    subgraph Behavior["Offline logging — rsi/controller.py"]
        BP["Behavior policy\nuniform-random n in {1,2,3,4}\npropensity = 0.25, recorded"]
        LOGSEEDS["BEHAVIOR_LOG_SEEDS\n201-210"]
        LOGS["LoggedRecord[]\n(seed, action_n, propensity,\nquality, cost_steps)"]
    end

    subgraph Estimate["Off-policy candidate proposal"]
        IPS["ips_estimate(logs, n)\nV_hat = (1/N) Σ 1{a=n}/propensity · quality\nuntried action ⇒ estimate = 0, never backfilled"]
        PICK["select_candidate\nargmax over n in {1,2,3,4}"]
    end

    subgraph Gate["Promotion gate — SELECTION_SEEDS 301-305, reused every cycle"]
        ONPOLICY_C["evaluate_on_policy(candidate_n)"]
        ONPOLICY_B["evaluate_on_policy(baseline_n)"]
        DECIDE{"candidate_quality ≥\nbaseline_quality?"}
    end

    subgraph Registry["PolicyRegistry"]
        ACTIVE["active_n\n(serves production requests)"]
        HIST["history: CycleRecord[]"]
        ROLLBACK["rollback()\nrevert to previous active_n"]
    end

    subgraph FinalEval["Final evaluation — FINAL_SEEDS 401-410, used once"]
        FCHECK["compare promoted policy\nvs original frozen baseline"]
        REPORT["sustained_improvement: bool\nnegative_transfer_detected: bool"]
    end

    LOGSEEDS --> BP --> LOGS --> IPS --> PICK
    PICK --> ONPOLICY_C
    ACTIVE -. "current baseline_n" .-> ONPOLICY_B
    ONPOLICY_C --> DECIDE
    ONPOLICY_B --> DECIDE
    DECIDE -- "yes: promote" --> ACTIVE
    DECIDE -- "no: keep baseline" --> ACTIVE
    ACTIVE --> HIST
    HIST -. "on regression" .-> ROLLBACK
    ROLLBACK --> ACTIVE
    ACTIVE --> FCHECK
    FCHECK --> REPORT

    MODEL -. "never updated\nthis milestone" .-> Gate
    EXECPOL -. "never updated\nthis milestone" .-> Gate

    classDef frozen fill:#f4f4f4,stroke:#888,stroke-dasharray: 4 3;
    class Frozen frozen;
```

**The load-bearing property**: `FINAL_SEEDS` never appears anywhere in
`Behavior`, `Estimate`, or `Gate` -- candidate search and promotion decisions
are made entirely without seeing the seeds the final claim is judged on
(`tests/test_rsi_controller.py::test_final_evaluation_uses_only_final_seeds_not_selection_or_behavior_seeds`).

## 3. Total design: charter layers vs. what's actually built

The charter's own architecture table (issue #5, section 2), annotated with
which JX milestone implements each layer and how completely.

```mermaid
flowchart LR
    subgraph L1["Model/sampler contract"]
        L1A["types.py: GenerationSpec, ModelRef, Budget"]
        L1B["jax_backend/model.py: TinyEpsilonModel"]
        L1C["jax_backend/sampler.py: DDPM, fold_in RNG discipline"]
    end

    subgraph L2["Compiled execution"]
        L2A["jax_backend/benchmark.py:\ncompile/dispatch/execute/transfer\nseparated & synchronized (JX-02)"]
    end

    subgraph L3["Kernel dispatch"]
        L3A["benchmarks/modal/jx03_gpu_internals.py\nelementwise fusion, tiled GEMM,\nNumPy correctness oracle (JX-03)"]
        L3B["⚠ Nsight Compute traces: NOT captured\n(no privileged perf-counter access)"]
    end

    subgraph L4["Distributed executor"]
        L4A["jax_backend/distributed.py\npmap, placement-invariant RNG,\ncollective latency sweep (JX-04)"]
        L4B["⚠ real multi-host, rank failure: NOT exercised"]
    end

    subgraph L5["Request scheduler"]
        L5A["scheduler.py: StepBatchScheduler (#2)"]
        L5B["jax_backend/serving.py:\nCompiledVariantCache, static vs\nchunk scheduling, p50/p95/p99 (JX-05)"]
    end

    subgraph L6["State store"]
        L6A["jax_backend/state.py: TrajectoryState,\nreal checkpoint/resume, integrity-checked (JX-01)"]
        L6B["checkpoints.py: ExactCheckpointStore (#2)"]
    end

    subgraph L7["Quality/search controller"]
        L7A["eval/quality_compute.py: frozen\nquality/compute protocol (JX-02)"]
        L7B["eval/adaptive_inference.py:\nsingle/more-steps/best-of-N/adaptive,\ndisjoint held-out eval (JX-06)"]
    end

    subgraph L8["Improvement controller"]
        L8A["rsi/controller.py: IPS off-policy\nestimation, promotion gate,\nPolicyRegistry.rollback() (JX-07)"]
        L8B["⚠ model-parameter training: NOT attempted"]
    end

    subgraph L9["Evidence layer"]
        L9A["reports/*.json: raw evidence per milestone"]
        L9B["docs/learning/*.md: findings,\nincluding kept negative results"]
    end

    subgraph L10["Not started"]
        L10A["JX-08: video workload,\nsecond accelerator qualification"]
    end

    L1 --> L2 --> L3 --> L4 --> L5 --> L6
    L6 --> L7 --> L8 --> L9
    L9 -.-> L10

    classDef done fill:#e8f4ea,stroke:#2f7a3d;
    classDef partial fill:#fff6e0,stroke:#b8860b;
    classDef todo fill:#fbe9e9,stroke:#b33;
    class L1,L2,L6,L9 done;
    class L3,L4,L5,L7,L8 partial;
    class L10 todo;
```

**Reading the color coding**: green layers have no known open gaps at their
current (tiny-model, single-host) scope. Amber layers work and are tested
but have an explicitly documented gap (no Nsight traces, no real multi-host,
no model-parameter training, etc.) -- the gap is named in the corresponding
`docs/learning/*.md` file, not implied to be closed. Red is simply not
started.
