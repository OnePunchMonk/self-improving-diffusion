# Research resources

Raw reading list behind `DESIGN.md` and `ARCHITECTURE.md`. Grouped by topic, roughly in the order it's useful to read them. No summaries beyond a one-line "why this one" — go read the actual papers.

## 1. Test-time scaling / search over diffusion trajectories

- [Inference-Time Scaling in Diffusion Models through Iterative Partial Refinement](https://arxiv.org/html/2605.19317v1) (ICLR 2026, [OpenReview](https://iclr.cc/virtual/2026/10018692)) — sequential-scaling baseline: refine, then verify, repeat.
- [SETS: Leveraging Self-Verification and Self-Correction for Improved Test-Time Scaling](https://arxiv.org/pdf/2501.19306) — self-verify without an external reward model.
- [VFScale: Intrinsic Reasoning through Verifier-Free Test-time Scalable Diffusion Model](https://arxiv.org/pdf/2502.01989) — uses the model's own energy/score function as the search signal instead of a learned verifier.
- [Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification for Discrete Diffusion Language Models](https://arxiv.org/html/2602.01842v3) — hierarchical search, efficiency-focused; discrete-diffusion but the search structure transfers.
- [Don't Scale It All: Training-Free Localized Test-Time Scaling for Diffusion Models](https://openreview.net/forum?id=ku4eCDtus3) — the load-bearing idea for `ARCHITECTURE.md`: spend extra compute only on the region that's actually bad, not the whole image.
- [ImageDoctor: Diagnosing Text-to-Image Generation via Grounded Image Reasoning](https://arxiv.org/pdf/2510.01010) — VLM-based critic that localizes failures; template for the discriminator+localizer in `verifiers/`.
- [dMLLM-TTS: Self-Verified and Efficient Test-Time Scaling for Diffusion Multi-Modal LLMs](https://arxiv.org/html/2512.19433) — in-loop self-verification without an external verifier, multimodal setting.
- [Test Time Scaling of Diffusion Model via Flow Matching Corrector](https://openreview.net/forum?id=80rSu6iYo0)

## 2. Preference optimization / RL for diffusion and video

- Diffusion-DPO (the base recipe: DPO loss via the ELBO as a log-likelihood proxy) — read this before any of the below.
- [VideoDPO: Omni-Preference Alignment for Video Diffusion Generation](https://arxiv.org/pdf/2412.14167) — first DPO applied to text-to-video; preference over visual quality + motion + text alignment jointly.
- [Align Video Diffusion Model with Online Video-Centric Preference Optimization](https://openaccess.thecvf.com/content/WACV2026/papers/Zhang_Align_Video_Diffusion_Model_with_Online_Video-Centric_Preference_Optimization_WACV_2026_paper.pdf) (WACV 2026)
- [SEE-DPO: Self Entropy Enhanced Direct Preference Optimization](https://arxiv.org/pdf/2411.04712)
- FlowGRPO, DanceGRPO, DRaFT, VADER — RL/policy-gradient reward maximization through the diffusion sampling chain (search these names directly; multiple concurrent papers, worth comparing).
- SPIN-Diffusion — self-play: current checkpoint's samples = "chosen," previous checkpoint's samples on the same prompt = "rejected." No human labels needed. This is the core mechanism behind the RSI framing.
- [ROCM: RLHF on Consistency Models](https://arxiv.org/html/2503.06171v1) — RLHF specifically for few-step/consistency-model students, relevant once you distill.

## 3. Failure modes — read before building the training loop, not after it breaks

- [Rethinking Direct Preference Optimization in Diffusion Models](https://arxiv.org/pdf/2505.18736)
- [Beyond Reward Margin: Rethinking and Resolving Likelihood Displacement in Diffusion Models via Video Generation](https://arxiv.org/html/2511.19049v1) — documents rejected-sample reward rising alongside chosen-sample reward (reward hacking signature); this is the exact thing `DESIGN.md` §5 says to guard against.
- [Process-based Self-Rewarding Language Models](https://arxiv.org/pdf/2503.03746) — self-rewarding loop failure modes and mitigations, LLM setting but the dynamics (reward drift, self-reinforcing errors) generalize directly to the RSI loop here.

## 4. Reward / critic modeling

- [Fake it till You Make it: Reward Modeling as Discriminative Prediction](https://arxiv.org/pdf/2506.13846) — discriminative (real-vs-fake) reward instead of scalar regression; more robust to reward hacking, motivates the `verifiers/` design.
- [Seedance 1.0: Exploring the Boundaries of Video Generation Models](https://arxiv.org/pdf/2506.09113) — industrial report; how a real lab structures reward-model/policy co-training at scale.

## 5. Inference-serving architecture (the SGLang side of `ARCHITECTURE.md`)

- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/pdf/2312.07104) — the original paper: RadixAttention, continuous batching, structured-generation DSL.
- [Fast and Expressive LLM Inference with RadixAttention and SGLang](https://www.lmsys.org/blog/2024-01-17-sglang/) (LMSYS blog) — most readable intro to RadixAttention specifically.
- [LLM Query Scheduling with Prefix Reuse and Latency Constraints](https://arxiv.org/pdf/2502.04677) — scheduling theory behind prefix-cache-aware batching, relevant to the step-cache scheduler design.

### Step/activation caching for diffusion (the closest existing analog to a "diffusion KV cache")
- TeaCache, MagCache — heuristic step-caching for diffusion/video generation based on output-difference thresholds (search these names; they're the direct precedent for `engine/`'s step-cache).
- [Budget-Constrained Step-Level Diffusion Caching](https://arxiv.org/html/2606.13496v1)
- [Accelerating Frequency Domain Diffusion Models with Error-Feedback Event-Driven Caching](https://arxiv.org/pdf/2604.22901)

### Diffusion *language* model serving (different modality, same KV-cache-incompatibility problem worth understanding)
- [dInfer: An Efficient Inference Framework for Diffusion Language Models](https://arxiv.org/pdf/2510.08666) — vicinity KV-cache refresh; useful analogy even though it's text, not image.
- [Affix Cache for Diffusion Large Language Models](https://arxiv.org/html/2608.26140)
- [Sangam: Efficiently Serving Diffusion LLMs with the AR Stack](https://arxiv.org/pdf/2607.04206)
- [DiLaServe: High SLO Attainment Serving for Diffusion Language Models](https://arxiv.org/pdf/2606.29094)
- [BlockBatch: Multi-Scale Consensus Decoding for Efficient Diffusion Language Model Inference](https://arxiv.org/pdf/2605.29233)
- [ObjectCache: Layerwise Object-Storage Retrieval for KV Cache Reuse](https://arxiv.org/pdf/2605.22850)
- [DEER: Draft with Diffusion, Verify with Autoregressive Models](https://arxiv.org/pdf/2512.15176)

## 6. Trackers — keep these bookmarked, actively updated

- [Awesome-RLHF-Video-Diffusion](https://github.com/wangqiang9/Awesome-RLHF-Video-Diffusion)

## Reading order if starting from zero

1. SGLang paper (§5) — understand the serving-framework shape being copied.
2. Diffusion-DPO + SPIN-Diffusion (§2) — understand the self-play preference mechanism.
3. "Don't Scale It All" + ImageDoctor (§1) — understand localized TTS and the critic-as-localizer idea.
4. The two failure-mode papers (§3) — know what breaks before you build the training loop.
5. TeaCache/MagCache + Budget-Constrained Step-Level Diffusion Caching (§5) — understand the step-cache precedent.
6. Everything else, as needed per component.

## Things to actually think about, not just read

Honest assessment, worth re-reading before sinking months into this:

**Is this a genuinely good, hard problem?** Yes, with a caveat. The RadixAttention analog is harder than the original — token-prefix cache hits are exact, activation-similarity cache hits for diffusion are fuzzy, and a wrong cache hit produces visibly wrong pixels, not a retryable wrong token. Step-level continuous batching across variable resolutions/step-counts has no clean "sequence length" bucket to fall back on. And RSI-in-the-serving-path (vs. RSI as a bolted-on training script) is the least explored part of all of it — the failure modes are documented (§3) but not solved. The caveat: this is currently a **systems research problem**, not a product problem. Nobody is blocked waiting for it the way people were blocked on LLM throughput in 2023 — the case for why it matters has to come from you, not from an already-agreeing market. Be honest about which of those you're optimizing for.

**Will building it actually teach RSI, diffusion, and inference — in equal measure? Probably not, by default.**
- *Inference/systems*: yes, hardest and most directly — the scheduler, the cache eviction/correctness policy, and batching logic are where the real novel difficulty lives, so it's where the most growth happens almost automatically.
- *Diffusion*: yes, but shallower than expected if you stay at the framework layer. You can build the whole `engine/` without ever wrestling with why diffusion models fail or how to improve sample quality — `runtime/` is deliberately a thin, swappable layer. Diffusion-modeling depth only comes from actually working in `verifiers/` and the training loop.
- *RSI*: the weakest link. Self-play DPO + threshold-triggered retraining is one narrow instance of self-improvement — closer to "automated preference-data flywheel" than to the harder RSI questions (does the loop compound over many rounds without collapsing, what's the ceiling, how do you detect plateau vs. silent degradation). At small scale the loop will likely just work for a few rounds, and the genuinely interesting failure modes only show up if you push it hard and long — which is also the part most likely to get deferred because "the framework isn't ready yet."

**The practical risk**: the engine/cache work is the most concrete, most satisfying-to-make-progress-on part of this project — and also the part least likely to teach you anything new about RSI specifically. If RSI is what you most care about, protect time for actually running the self-play loop for many rounds and staring at whether it's compounding, not just for building the thing that serves it.
