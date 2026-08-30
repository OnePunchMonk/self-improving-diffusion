# Self-Improving Inference Systems for Diffusion, Vision & Video Models

Research + design notes. Private working repo — not for publication as-is.

## 1. What "self-improving inference" means here

Three distinct mechanisms get conflated under this name; keep them separate because they compose differently:

1. **Test-time scaling (TTS)** — spend more compute per sample at inference to get a better output, no weight updates. Parallel (best-of-N + verifier/rerank) or sequential (iterative refinement / self-correction).
2. **Online/continual weight improvement** — use signal generated during inference (verifier scores, preference pairs, self-play) to update the model itself, so tomorrow's inference is better without a human labeling round.
3. **Distillation loops** — a slow/expensive teacher process (large model + search + verifier) produces training data that a fast student model is distilled into, shrinking the TTS cost over generations.

An interesting system usually chains these: TTS produces high-quality samples → those become preference/SFT data → periodic fine-tune → the improved model needs less TTS to hit the same quality bar next round. That compounding loop *is* the "self-improving" part; a single verifier-guided sampler alone is just TTS.

## 2. Landscape (what to read, grouped by mechanism)

### Test-time scaling / search over diffusion trajectories
- **Inference-Time Scaling in Diffusion Models through Iterative Partial Refinement** (ICLR 2026) — sequential scaling, refine-then-verify loop.
- **SETS: Self-Verification and Self-Correction for TTS** (arXiv 2501.19306) — self-verify without an external reward model.
- **VFScale: Verifier-Free Test-time Scalable Diffusion** (arXiv 2502.01989) — uses the model's own energy/score function as the verifier signal — important because it avoids reward-model overfitting/hacking.
- **Prism** (arXiv 2602.01842) — hierarchical search + self-verification, efficiency-focused.
- **ImageDoctor** (arXiv 2510.01010) — grounded VLM-based diagnosis of T2I failures; a good template for building a *critic* rather than a scalar reward model.
- **"Don't Scale It All"** (OpenReview ku4eCDtus3) — training-free *localized* TTS: spend extra compute only on the region of the image/frame that's actually bad. This is the most practically useful idea for video, where full-sequence resampling is expensive.

### Preference optimization / RL for diffusion & video
- **Diffusion-DPO** — DPO loss via the ELBO as a log-likelihood proxy; the base recipe everything below extends.
- **VideoDPO** (arXiv 2412.14167) — first DPO applied to text-to-video, omni-preference (visual quality + motion + text alignment).
- **FlowGRPO, DanceGRPO, DRaFT, VADER** — RL/policy-gradient style reward maximization directly through the diffusion sampling chain.
- **SPIN-Diffusion** and **SEE-DPO** — *self-play*: the model's own earlier checkpoint's generations become the "rejected" sample and current-checkpoint generations become "chosen," removing the need for human preference labels entirely. This is the cleanest self-improvement loop in the list.
- **"Rethinking DPO in Diffusion Models"** (arXiv 2505.18736) and the **likelihood displacement** paper (arXiv 2511.19049) — both document a real failure mode: naive diffusion-DPO can raise reward on preferred samples while *also* raising it on some rejected ones (reward hacking / distribution collapse). Read before building the RL loop, not after it breaks.
- **ROCM** (arXiv 2503.06171) — RLHF specifically on consistency/few-step models, relevant if the eventual student is a fast sampler.

### Reward/critic modeling
- **"Fake it till You Make It"** (arXiv 2506.13846) — reward modeling as discriminative prediction (real vs. generated) instead of learned scalar reward, cheaper and more robust to reward hacking.
- Seedance 1.0 report (arXiv 2506.09113) — an industrial video model report; useful for how a real lab structures the iterative reward-model-and-policy co-training loop at scale.

### Trackers / running bibliographies
- [Awesome-RLHF-Video-Diffusion](https://github.com/wangqiang9/Awesome-RLHF-Video-Diffusion) — keep this starred, it's actively updated.

## 3. Where to actually learn this (beyond papers)

- **Read the training code, not just the paper.** Diffusers' `examples/` (dpo, reward finetuning), and the open reimplementations linked from the papers above, show the parts papers omit: how KL regularization is scheduled, how preference pairs are batched against the noise schedule, gradient checkpointing through the sampler for RL.
- **Build the verifier before the policy.** Any of TTS/DPO/RL is only as good as the critic. Start by building/eval'ing an ImageDoctor-style VLM critic against a small human-labeled set; measure its agreement with humans before trusting it as a reward.
- **Small-scale reproduction first.** Reproduce SPIN-Diffusion or Diffusion-DPO on a tiny SD1.5/consistency-model checkpoint on a single GPU before touching video — video multiplies every cost (frames × search width × verifier calls) and hides bugs in the same loop that are cheap to find on images.
- **Communities/venues:** ICLR/NeurIPS/CVPR workshops on generative models and RLHF-for-diffusion specifically (there's now a dedicated one most cycles); the arXiv "cs.CV" + "cs.LG" cross-list scan for "diffusion DPO" / "test-time scaling" weekly is more current than any course.

## 4. Design: an interesting way to build this

**Core idea: a closed loop with three sizes of the "same" model, connected by localized test-time search and self-play preference data — image-first, video by frame-chunking.**

```
                 ┌────────────────────────────────────────┐
                 │              CRITIC (VLM)                │
                 │  scores + localizes failures per-region  │
                 └───────────────┬───────────────┬─────────┘
                                  │ score          │ region mask
                                  ▼                ▼
   prompt ──► POLICY (diffusion) ──► sample ──► localized refinement
                  ▲   (few-step,                 (resample only bad
                  │    student)                   regions — cheap TTS)
                  │                                    │
                  │                                    ▼
                  │                          accepted sample = "chosen"
                  │                          earlier-checkpoint sample
                  │                          on same prompt = "rejected"
                  │                                    │
                  └──────── periodic DPO/self-play ◄────┘
                          fine-tune (student weights)
```

Why this shape, concretely:

1. **Localized TTS as the inference-time engine.** Instead of best-of-N over whole images/clips (expensive, especially for video), use the critic to point at *which region/frame* is bad and only re-sample that part (per "Don't Scale It All"). For video this is the difference between resampling 1 frame vs. 120.
2. **Self-play instead of human preference data for the training signal.** Following SPIN-Diffusion/SEE-DPO: current policy's accepted (post-refinement) sample vs. last-epoch checkpoint's sample on the same prompt = a free preference pair. No human labeling loop required to keep the system improving.
3. **Critic built as a discriminator, not a scalar reward regressor** (per the "Fake it till you make it" result) — more robust to reward hacking, and doubles as the localization signal for step 1 by looking at per-patch/per-frame real-vs-fake logits (interpretable as a rough saliency map for "where is this fake").
4. **Explicitly guard against the known failure mode**: track reward on a held-out prompt set and watch for the "rejected samples' reward also rising" signature from the likelihood-displacement paper — that's the trigger to freeze fine-tuning and inspect data, not a thing to discover in prod.
5. **Distill down after N self-play rounds.** Once the policy + localized-TTS loop plateaus at a given quality, distill into a smaller/fewer-step student (consistency-model style, cf. ROCM) so the *next* self-play round starts from a cheaper base — this is what makes the loop compound rather than just oscillate at fixed cost.
6. **Video = frame-chunked reuse of the same machinery**, not a separate system: run the image-level loop per keyframe, propagate corrections through a lightweight temporal-consistency pass, only invoke full-sequence critic scoring sparingly (it's the most expensive VLM call).

### Suggested build order (repo scaffold reflects this)
1. `src/verifiers/` — discriminative critic (real/fake + localization), evaluated against human labels before anything else trusts it.
2. `src/search/` — localized best-of-N / iterative refinement using the critic's region mask.
3. `src/data/` — self-play preference pair construction (current vs. last-checkpoint samples on shared prompts).
4. `src/training/` — DPO/self-play fine-tuning step, with the reward-hacking guard from (4) above as a hard training-loop check, not just a metric.
5. `src/eval/` — held-out prompt reward tracking + human-agreement checks on the critic, run every round.
6. Video: extend `search/` and `verifiers/` with a keyframe/temporal-consistency layer once the image loop is stable — do not start with video.

## 5. Open risks worth tracking from day one
- Reward hacking / distribution collapse in self-play DPO (Section 2's DPO papers) — needs the held-out check in `eval/` from round 1.
- Critic drift: a discriminator trained against round-1 policy samples degrades as the policy improves ("moving target") — plan to periodically refresh the critic's negative set with newer policy samples.
- Video cost blowup: localized TTS + sparse full-sequence critic calls are load-bearing for keeping this affordable; don't skip that design point to "get video working first."
