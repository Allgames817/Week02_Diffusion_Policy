# 01 — Diffusion Policy Paper Concepts

Companion to [README.md](README.md). Focus: ideas from *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion* (Chi et al.) as they map to this Week 2 **lowdim PushT** run and to leftovers from [Week 1 ACT](https://github.com/Allgames817/Week01_ACT).

---

## 1. From ACT leftovers to diffusion

Week 1 ended with open questions:

1. How to model **multimodal** action distributions better?
2. Is **CVAE** the best action generator here?
3. Is **train-$z$ / infer-$z{=}0$** mismatch ideal?
4. How to generate smoother, more robust action trajectories?

Diffusion Policy addresses these by treating the action chunk as the object of a **conditional generative model**:

$$
p_\theta(A_t \mid O_t)
$$

Training learns to predict the noise that was added to clean expert actions. Inference samples from noise with a DDPM or DDIM scheduler. There is **no** separate CVAE encoder that is dropped at test time.

This note only covers the **low-dimensional UNet** setup used locally (`DiffusionUnetLowdimPolicy`). Image, hybrid, and Transformer diffusion variants exist in the official repo and were **not** trained this week.

---

## 2. Behavior cloning with a generative action head

Expert demonstrations still provide supervised targets. The difference from single-step MSE BC (and from ACT’s L1 + KL) is the training target:

$$
\tilde{A} = \sqrt{\bar\alpha_t}\, A + \sqrt{1-\bar\alpha_t}\,\epsilon,\qquad
\mathcal{L} = \mathbb{E}\big[\|\epsilon - \epsilon_\theta(\tilde{A}, t, O)\|_2^2\big]
$$

| Piece | Role this week |
|---|---|
| $A$ | clean action window `[B, 16, 2]` |
| $O$ | first `n_obs_steps=2` observations → flattened `[B, 40]` |
| $t$ | discrete noise level drawn during training |
| $\epsilon_\theta$ | CondUnet1D |

Closed-loop **score** (episode max reward, then averaged) is an evaluation metric, not the training label.

---

## 3. Action chunking (shared idea with ACT)

Both ACT and Diffusion Policy predict a short action sequence rather than a single step. In this run:

| Quantity | Value | Notes |
|---|---|---|
| Prediction window $T_p$ | 16 | stored / denoised length |
| Observation window $T_o$ | 2 | conditioning only |
| Execution window $T_a$ | 8 | after temporal alignment slice |

Week 1 showed that chunk length $k$ strongly affects Transfer Cube success. Week 2 keeps $(T_p, T_o, T_a)$ **fixed** and instead varies **inference compute** $K$ (number of denoising steps). These are different levers:

- Chunk length → how much future action is planned per query.
- $K$ → how much compute is spent to sample one chunk from the diffusion model.

---

## 4. Multimodality without CVAE $z$

ACT models multimodality with a latent $z$ (train sample / infer zero). Diffusion models multimodality by **stochastic sampling**: different initial noises (and DDIM $\eta$ if nonzero) can yield different action trajectories for the same observation.

This week:

- DDIM uses **$\eta = 0$** (deterministic updates given the initial noise).
- Initial noise is still random; **sampling seeds** change trajectories.
- Empirically, env `100003` + sampling seed `0` fails for DDPM and DDIM-20 while other seeds on the same env do not (see [05_Behavior_Case.md](05_Behavior_Case.md)).

So multimodality / sampling noise remains visible even with $\eta=0$.

---

## 5. Receding-horizon control

At runtime the policy is queried repeatedly. Each query denoises a full horizon, then only $T_a$ actions are executed before the next observation is taken. This is analogous in spirit to ACT’s closed-loop querying (with or without temporal aggregation), but the **generator** inside each query is a diffusion sampler rather than a Transformer decoder + optional $z$.

EMA in Diffusion Policy averages **weights** across training steps. It is **not** ACT’s exponential average of overlapping predicted actions (temporal aggregation).

---

## 6. Why ablate inference steps?

DDPM training uses $N{=}100$ noise levels. At test time, DDIM can reuse the same trained $\beta$ schedule with fewer steps $K$ that divide $N$, without retraining. The paper and engineering practice care about the **speed–quality** tradeoff.

Week 2 Day 6 measures that tradeoff on one self-trained checkpoint:

| $K$ (DDIM) | mean score (5×3) | median latency | vs DDIM-100 |
|---|---:|---:|---:|
| 100 | 0.991 | 1116.1 ms | 1.00× |
| 50 | 0.996 | 557.1 ms | 2.00× |
| 20 | 0.935 | 231.9 ms | 4.81× |
| 10 | 0.987 | 116.2 ms | 9.60× |

Full protocol and caveats: [06_Inference_Step_Ablation.md](06_Inference_Step_Ablation.md).

---

## 7. What this week does **not** claim

- Full paper benchmark numbers (different env counts, vision settings, training length).
- That Diffusion dominates ACT on Transfer Cube (different task / embodiment this week).
- That $K{=}10$ is universally optimal.
- That simulation latency equals real-robot closed-loop stability.
