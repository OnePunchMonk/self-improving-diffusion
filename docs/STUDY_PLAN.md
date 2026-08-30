# Study plan: three tracks

Separate from `RESEARCH.md` (which is project-specific, about this repo). This is the broader curriculum for the three things you actually care about — read/watch/run these regardless of where the repo itself ends up. Overlaps with `RESEARCH.md` are noted, not duplicated in full.

---

## Track 1 — Inference frameworks

**Docs to actually read line-by-line (not just skim):**
- [SGLang docs](https://docs.sglang.ai/) — read the scheduler and RadixAttention pages before touching code.
- [vLLM docs](https://docs.vllm.ai/) — PagedAttention design doc specifically.
- vLLM's own [PagedAttention blog post](https://blog.vllm.ai/2023/06/20/vllm.html) — the original announcement, still the clearest single explanation of the memory-fragmentation problem it solves.
- TensorRT-LLM docs — read for contrast: it's an AOT-compiled engine, not a Python scheduler, and seeing why that tradeoff exists clarifies what SGLang/vLLM are trading away for flexibility.

**Papers:**
- [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/pdf/2312.07104) — already in RESEARCH.md §5, read it here first if you haven't.
- Orca (continuous batching, the paper that predates and motivated vLLM/SGLang's schedulers) — search "Orca: A Distributed Serving System for Transformer-Based Generative Models."
- PagedAttention paper (vLLM) — the systems argument for treating KV cache like OS virtual memory pages.
- [LLM Query Scheduling with Prefix Reuse and Latency Constraints](https://arxiv.org/pdf/2502.04677) — scheduling theory, already in RESEARCH.md.

**Repos to actually read source, not just `pip install`:**
- `sgl-project/sglang` — read `scheduler.py` and the radix cache implementation before anything else.
- `vllm-project/vllm` — read the `core/block_manager.py` (paged memory) and the scheduler.
- Comparison writeups for orientation, not depth: [vLLM vs SGLang vs TensorRT-LLM](https://jarvislabs.ai/blog/vllm-sglang-trtllm-comparison), [Best LLM Inference Engines 2026](https://www.yottalabs.ai/post/best-llm-inference-engines-in-2026-vllm-tensorrt-llm-tgi-and-sglang-compared).

**Exercise, not just reading:** trace one request end-to-end through SGLang's scheduler by reading source with a debugger attached — the "aha" for how continuous batching actually works comes from watching slots free and refill in real time, not from the diagram in the paper.

---

## Track 2 — RSI (recursive self-improvement)

**Start here:**
- [Anthropic's RSI report](https://www.mindstudio.ai/blog/what-is-recursive-self-improvement-ai-anthropic-rsi-report) (or find Anthropic's original directly) — what RSI actually looks like, how close we are, what responsible development requires. This is the most level-headed thing in the space; read it before the hype pieces.
- [Recursive Self-Improvement in AI: From Bounded Self-Refinement to Autonomous Research Loops](https://arxiv.org/html/2607.07663v1) — a survey of ~1,250 arXiv papers (2024–2026), organized by *what the system improves* and *degree of loop closure* (human-in-the-loop → fully closed). This is the single best map of the field; use it to navigate rather than reading linearly.
- [awesome-rsi](https://github.com/lobehub/awesome-rsi) — curated map of models, agents, harnesses, automated-AI-R&D work, benchmarks, and safety considerations. Treat as a living index, revisit monthly.

**Concrete implementations worth studying (not just citing):**
- AIDE² (Weco AI) — [blog writeup](https://www.weco.ai/blog/first-evidence-of-recursive-self-improvement): an RSI system that autonomously redesigned a search algorithm and cut prompt size 16× over 8 days. Read for the harness design (what closes the loop, what gates each iteration), not the result.
- Frontis-MA1, FT-Dojo, MLEvolve, The AI Scientist-v2 — search these names directly; 2026 implementations of automated-research loops, useful for seeing different loop-closure strategies in practice.

**Adjacent surveys** (each covers a different slice of "self-evolving" systems — skim all three, they overlap but each has unique framing):
- "A Survey of Self-Evolving Agents: On Path to Artificial Super Intelligence"
- "A Comprehensive Survey of Self-Evolving AI Agents"
- "A Survey on Self-Evolution of Large Language Models"

**Venue to watch:** [ICLR 2026 Workshop on AI with Recursive Self-Improvement](https://iclr.cc/virtual/2026/workshop/10000796) — check the accepted papers list once posted, it'll be the highest-signal single source for what the field considers open.

**Commentary, for calibration not conclusions:** Matt Shumer's "Something Big Is Happening" (Feb 2026) — read as a data point on how the idea is landing publicly, not as a technical source. Cross-check any claim in it against the survey above before believing it.

**The RSI-specific mechanism relevant to *this* project** — self-play preference loops (SPIN-Diffusion) and their failure modes (reward hacking / likelihood displacement, RESEARCH.md §3) — is a narrow, bounded instance of RSI (single model, single metric, one type of loop closure). Read the broader survey specifically to see how much wider the design space is, so you don't mistake "I built a self-play DPO loop" for "I understand RSI."

---

## Track 3 — Diffusion models

**Blogs, in order:**
1. [Lil'Log — What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) — the standard first read, still the best single overview connecting DDPM/score-matching/SDE views.
2. Yang Song's blog on score-based generative modeling through SDEs — the deeper mathematical treatment behind Lil'Log's summary; read after, not instead of.
3. [The Annotated Diffusion Model](https://huggingface.co/blog/annotated-diffusion) (Hugging Face) — line-by-line PyTorch DDPM implementation. Actually run it, don't just read it.
4. Alex Alemi's blog on Variational Diffusion Loss — for the ELBO derivation once you want to understand where Diffusion-DPO's log-likelihood proxy (RESEARCH.md §2) comes from.

**Foundational papers, in order:**
1. DDPM (Ho et al., 2020) — the paper that started the current wave.
2. NCSN (Song & Ermon, 2019) — score-based framing, predates and is unified with DDPM by the SDE view.
3. Score-based SDE paper (Song et al.) — unifies DDPM and NCSN as discretizations of forward/reverse SDEs; this is the paper that makes "diffusion" and "score-based" the same thing in your head.
4. DDIM — deterministic sampling, the basis for every fast/few-step sampler since.
5. Classifier-free guidance — the mechanism every prompt-conditioned model you'll touch actually uses at inference.
6. Latent diffusion (Stable Diffusion's paper) — why almost nothing runs in pixel space anymore, directly relevant to what `runtime/` in this repo will actually be doing.

**Repos to study, not just use:**
- `huggingface/diffusers` — read the scheduler implementations (`DDIMScheduler`, `DPMSolverMultistepScheduler`) to see how the SDE/ODE math from the papers above becomes actual step code. This is the single most useful repo for connecting theory to what `engine/`'s step-cache will need to key on.
- [mbreuss/diffusion-literature-for-robotics](https://github.com/mbreuss/diffusion-literature-for-robotics) — despite the name, a well-maintained curated list of diffusion papers/blogs beyond robotics; good for filling gaps.
- DPM-Solver repo — the ODE-solver work behind fast sampling; relevant once you care about step-count reduction (which interacts directly with the step-cache design).

**Sampler/serving-adjacent, ties into Track 1:**
- TeaCache, MagCache repos (RESEARCH.md §5) — read the actual caching heuristic code, it's small and directly informs `engine/`'s step-cache design.

---

## Track 4 — Harness, eval, and post-training

This is the connective tissue between Track 2 (RSI) and this project's actual `rsi/training` + `eval/` components — the mechanics of *how* a model gets updated from generated data, and how you know it's actually gotten better rather than just different.

**Post-training algorithms, in order:**
- [Hugging Face: A Guide to RL Post-Training for LLMs — PPO, DPO, GRPO, and Beyond](https://huggingface.co/blog/karina-zadorozhny/guide-to-llm-post-training-algorithms) — the single best overview connecting all three algorithm families; includes real pitfalls (KL-divergence gradient estimation bugs in popular libraries), not just the clean-room math.
- [Reinforcement Learning for LLM Post-Training: A Survey](https://arxiv.org/pdf/2407.16216) — broader survey, use to see how PPO/DPO/GRPO fit into the wider post-training space (SFT, RLHF, RLVR).
- [Post-Training in 2026: GRPO, DAPO, RLVR & Beyond](https://llm-stats.com/blog/research/post-training-techniques-2026) — current-state overview of where the field moved past vanilla PPO/DPO.
- Why GRPO matters for this project specifically: it drops the value network and estimates baselines from group-level statistics over sampled responses — directly applicable to the self-play group of samples this repo's `rsi/data` produces per prompt, worth comparing against plain pairwise DPO.
- [Relax: An Asynchronous Reinforcement Learning Engine for Omni-Modal Post-Training at Scale](https://arxiv.org/pdf/2604.11554) — an actual *engine* for post-training (not just an algorithm), the systems-side analog of Track 1 applied to training instead of inference. Directly relevant to how `rsi/training` should be triggered asynchronously off the serving path.

**Harness design** (the code that runs the loop, not the algorithm):
- AIDE² and the other 2026 RSI implementations from Track 2 — reread specifically for harness structure: what gates each iteration, what's checked before accepting an update, how failures are caught. A harness is a large fraction of what actually makes an RSI loop safe to run unattended.
- The failure-mode papers in RESEARCH.md §3 (likelihood displacement, DPO rethinking) are effectively *eval harness requirements in disguise* — each documented failure mode is a check your harness needs before it ships an update.

**Eval, specifically for a self-improving loop:**
- The core problem eval has to solve here that a normal benchmark doesn't: distinguishing "the model actually got better" from "the model started gaming the reward/critic it's being trained against." That's the reward-hacking detection problem, not a generic eval-suite problem — held-out prompts scored by a *frozen* critic checkpoint (never updated alongside the policy) is the minimum viable defense.
- [LLM Eval vs RLHF Feedback Loops in 2026](https://futureagi.com/blog/llm-eval-vs-rlhf-feedback-2026/) — on why eval and the training reward signal need to stay decoupled, exactly the frozen-critic point above from a different angle.
- Practical exercise: before building anything else in `eval/`, write down the three ways this project's specific self-play loop could look like it's improving while actually collapsing (mode collapse, reward hacking the discriminator, critic drift) and build one check per failure mode. Don't build generic eval infrastructure first.

---

## Track 5 — GPU internals

The layer underneath both Track 1 (inference) and Track 4 (post-training engines) — you can't reason about step-cache cost, batching efficiency, or async training-job scheduling without knowing what's actually cheap and expensive on the hardware.

**Foundational:**
- *Programming Massively Parallel Processors* (Kirk & Hwu) — the standard textbook; work through the memory-coalescing and shared-memory chapters closely, skim the rest on a first pass.
- [Inside the GPU: A Guide to Modern Graphics Architecture](https://learnopencv.com/modern-gpu-architecture-explained/) — SM layout, CUDA vs. Tensor vs. RT cores, good visual orientation before the denser papers.
- [NVIDIA GPU Architecture Explained: A Guide for AI](https://www.thundercompute.com/blog/nvidia-gpu-architecture-explained) — SM → tensor core → memory hierarchy walkthrough across Ampere/Hopper/Blackwell; read for the generational deltas, since kernel-level tricks (e.g. what fits in shared memory) change across them.

**Memory hierarchy and where the actual bottlenecks live** (directly relevant to step-caching design in this project — a cache is only worth it if it beats a memory-bound recompute):
- [Can Tensor Cores Benefit Memory-Bound Kernels? (No!)](https://arxiv.org/pdf/2502.16851) — read this specifically before assuming caching intermediate activations is free; it isn't, if the recompute was already memory-bound rather than compute-bound.
- [Dissecting Tensor Cores via Microbenchmarks](https://arxiv.org/pdf/2206.02874) — real measured latency/throughput of tensor core ops, not vendor marketing numbers.

**Applied, ties directly into `runtime/`:**
- [Advanced GPU Optimization: Tensor Core Programming](https://dev.to/javadinteger/advanced-gpu-optimization-tensor-core-programming-nvidia-3708) — WMMA usage, PTX-level MMA instructions, tile-shape selection, mixed precision. This is the level `runtime/`'s denoiser backends will eventually need to be tuned at.
- FlashAttention paper (not in the search results above, but essential) — the canonical example of "know the memory hierarchy, restructure the algorithm around it" applied to exactly the kind of kernel diffusion models depend on (attention inside DiT blocks).

**Exercise, not just reading:** profile one denoising step of an actual diffusion model forward pass (Nsight Compute or even `torch.profiler`) and identify which parts are compute-bound vs. memory-bound before designing the step-cache — this project's central caching bet only pays off on the memory-bound parts, and you won't know which parts those are without measuring.

---

## How the tracks actually connect (so this doesn't feel like three separate reading lists)

- Track 3's scheduler math (DDIM/DPM-Solver step structure) is exactly what Track 1's step-cache needs a similarity key for.
- Track 1's "structured generation as a program" idea is what makes Track 3's critic-guided refinement composable instead of glued-together client code.
- Track 2's loop-closure taxonomy is the right lens for being honest about how much RSI this project's self-play DPO loop actually contains (see the caveat at the end of Track 2) — reread it after a few months of building and re-rate yourself.
- Track 4's eval-decoupling requirement and Track 2's failure-mode literature are the same problem from two angles: a self-improving loop's biggest risk isn't the algorithm, it's not noticing collapse.
- Track 5 is the reality check under Track 1's whole thesis: the step-cache is only worth building where the underlying compute is memory-bound, not compute-bound — measure before designing.
