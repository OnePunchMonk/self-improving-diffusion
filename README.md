# self-improving-diffusion

Research + prototyping for self-improving inference systems around diffusion, vision, and video models: test-time scaling, self-play preference optimization, and distillation loops that compound quality over rounds.

See `docs/DESIGN.md` for the literature review and system design.

## Layout
- `docs/` — design notes, literature review
- `src/verifiers/` — critic / reward models
- `src/search/` — test-time scaling / localized refinement
- `src/data/` — preference pair construction (self-play)
- `src/training/` — DPO / RL fine-tuning loops
- `src/eval/` — reward tracking, critic-human agreement checks
- `experiments/` — run configs and logs
- `notebooks/` — exploratory analysis
