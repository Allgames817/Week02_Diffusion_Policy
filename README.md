# Week 2 — Diffusion Policy

Research note for VLA Roadmap Week 2. Numbers come from a local PushT lowdim learning run and a controlled DDIM inference-step ablation on that checkpoint (copies under [`results/`](results/)). This is **not** a full paper reproduction.

**Detail docs:** [01 Paper](01_DP_Paper.md) · [02 Architecture](02_DP_Architecture.md) · [03 Code Map](03_DP_Code_Map.md) · [04 Reproduction](04_Reproduction.md) · [05 Behavior Case](05_Behavior_Case.md) · [06 Inference Ablation](06_Inference_Step_Ablation.md) · [Source Index](results/SOURCE_INDEX.md)

Chinese summary of the ablation table: [`results/CONCLUSION_zh.md`](results/CONCLUSION_zh.md) · [`results/comparison_table.md`](results/comparison_table.md).

---

## 1. Goal

- Connect Week 1 ACT leftovers (multimodality, CVAE train–infer mismatch, action generation) to **conditional diffusion over action chunks**.
- Train a local **DiffusionUnetLowdimPolicy** on PushT keypoints (20-D obs, 2-D action).
- Reload the checkpoint in a new process and evaluate closed-loop rollouts.
- Ablate **DDIM inference steps** $K \in \{100,50,20,10\}$ on one fixed model, with a full-chain **DDPM** reference.

---

## 2. Problem Formulation

**Single-step BC:**

$$
\pi(a_t \mid o_t)
$$

**ACT (action chunking + CVAE):**

$$
\pi(a_{t:t+k-1} \mid I_t, q_t),\quad z\sim q(z\mid a,q)\ \text{(train)},\quad z=0\ \text{(infer)}
$$

**Diffusion Policy (this week, lowdim):**

$$
p_\theta(A_t \mid O_t),\quad A_t = a_{t:t+T_p-1},\quad O_t = o_{t-T_o+1:t}
$$

Denoising at inference starts from Gaussian noise and runs $K$ scheduler steps conditioned on $O_t$.

| Symbol | Meaning (this run) |
|---|---|
| $o$ | keypoints observation, dim **20** |
| $a$ | 2-D agent position action |
| $T_p$ / `horizon` | **16** |
| $T_o$ / `n_obs_steps` | **2** → condition length **40** after flatten |
| $T_a$ / `n_action_steps` | **8** executed after temporal alignment |
| $N$ | training noise levels **100** |
| $K$ | inference denoising steps (ablated) |

---

## 3. Why Diffusion After ACT

Week 1 left open: multimodal actions, whether CVAE + $z{=}0$ is ideal, and how to generate smoother trajectories. Diffusion answers by learning a **noise-prediction** model over action sequences and sampling at test time — no separate train-$z$ / infer-$z{=}0$ path.

This week’s evidence is a **learning experiment** on PushT (not Transfer Cube): the same trained model supports DDIM with fewer steps and still keeps high mean score on a small eval panel. See [01_DP_Paper.md](01_DP_Paper.md) and [06_Inference_Step_Ablation.md](06_Inference_Step_Ablation.md).

![Speed–quality](figures/speed_quality.png)

---

## 4. Architecture (lowdim UNet used this week)

```
obs [B, 2, 20] → flatten → global_cond [B, 40]
clean actions [B, 16, 2]
 → add noise at random t ∈ {0..N-1}
 → CondUnet1D predicts noise
 → MSE(pred_noise, true_noise)

Inference:
 x_T ~ N(0,I)
 → K DDPM or DDIM updates conditioned on global_cond
 → unnormalize → slice action_pred[:, 1:9] → execute
```

EMA smooths **network weights** during training (different from ACT temporal aggregation of overlapping action chunks). Details: [02_DP_Architecture.md](02_DP_Architecture.md).

---

## 5. Code Map

```
train.py
 → TrainDiffusionUnetLowdimWorkspace
 → PushTLowdimDataset / DataLoader / normalizer
 → DiffusionUnetLowdimPolicy.compute_loss
 → EMA + checkpoint latest.ckpt
eval.py / local ablation runner
 → load ema_model
 → PushTKeypointsRunner (closed-loop)
 → mean of episode max rewards
```

Full map: [03_DP_Code_Map.md](03_DP_Code_Map.md). Scripts live in the local `diffusion_policy` checkout; this note does not vendor the upstream repo.

---

## 6. Reproduction (training)

Task: **PushT** lowdim keypoints. Local Hydra config inherits official UNet widths (`train_diffusion_unet_lowdim_workspace`). Machine: Windows + RTX 4060 Laptop, PyTorch 2.1.2+cu121, diffusers 0.11.1. Training seed **42**, **100** epochs, batch **32**, EMA on. Checkpoint SHA-256 `922e11a2…0001d` (full hash in [SOURCE_INDEX](results/SOURCE_INDEX.md)).

| Setting | Value |
|---|---|
| Final epoch train loss | **0.0124** (epoch 99) |
| Final epoch val loss | **0.0865** |
| Training-time test mean score (epoch 50, 3 inits) | **0.994** |
| New-process `final_eval` test mean score (3 inits) | **1.0** |
| `final_eval` train init mean score (1 init) | **0.980** |

Mean score = average of **episode max reward**, not binary success rate. Val loss rises late; `latest.ckpt` is last epoch, not score-best. Curves: [figures/train_val_loss.png](figures/train_val_loss.png), [figures/rollout_score.png](figures/rollout_score.png). Full write-up: [04_Reproduction.md](04_Reproduction.md).

---

## 7. Behavior Case

Under the ablation seeds, env **100000** / sampling seed **0** reaches max reward **1.0** for all five sampler settings. Env **100003** / sampling seed **0** fails for **DDPM-100** and **DDIM-20** (max reward **0.179**, full 300 steps) while DDIM-100/50/10 succeed. Videos: [`results/videos/`](results/videos/). Details: [05_Behavior_Case.md](05_Behavior_Case.md).

---

## 8. Inference Step Ablation

**Fixed:** same self-trained EMA checkpoint, normalizer, windows, beta schedule, DDIM $\eta{=}0$, leading spacing, env seeds `100000–100004`, sampling seeds `0,1,2`, max 300 env steps.

**Main independent variable:** DDIM $K \in \{100,50,20,10\}$. Full-chain DDPM is a **separate sampler reference** (not in the DDIM speedup denominator).

| Condition | Horizon of denoising | mean score | median ms | speedup vs DDIM-100 |
|---|---|---:|---:|---:|
| DDPM-100 (ref.) | full chain | 0.929 | 1458.7 | — |
| DDIM-100 | 100 | 0.991 | 1116.1 | 1.00× |
| DDIM-50 | 50 | 0.996 | 557.1 | 2.00× |
| DDIM-20 | 20 | 0.935 | 231.9 | 4.81× |
| DDIM-10 | 10 | 0.987 | 116.2 | 9.60× |

Source: [`results/summary.csv`](results/summary.csv) (15 rollouts per cell). Leading rule changes the first noise level (e.g. $K{=}10$ starts at level 90). Timing = CUDA-synced batch-1 `predict_action` on a fixed GPU observation.

**Candidate tradeoff on this panel:** $K{=}10$ (~9.60× vs DDIM-100, paired score delta −0.004). Not a universal optimum.

![Score vs steps](figures/score_vs_steps.png)

![Latency vs steps](figures/latency_vs_steps.png)

Full write-up: [06_Inference_Step_Ablation.md](06_Inference_Step_Ablation.md).

---

## 9. Main Findings

1. **Conditional diffusion over action chunks** trains and reloads successfully on local PushT lowdim (learning-scale eval).
2. **Offline noise-prediction loss ≠ closed-loop score** in the same sense as Week 1: val loss rose late while rollout scores stayed high on the small panel.
3. **Prediction horizon (action window)** and **inference compute ($K$)** are different levers; reducing $K$ with DDIM does not require retraining.
4. On this checkpoint, **DDIM-10** kept mean score near DDIM-100 while cutting median latency by ~9.6×.
5. **Mean score can hide sampler-specific failures** (env 100003 / noise 0 for DDPM and DDIM-20).

---

## 10. What This Week Leaves Open

1. Image / hybrid / transformer Diffusion Policy variants.
2. Larger eval panels and true success-rate reporting.
3. Whether fewer **training** noise levels ($N$) need a new model.
4. Real-robot latency and closed-loop stability under compute delay.
5. Direct ACT vs Diffusion head-to-head on the same task.

→ **Next week** can pick a vision-conditioned policy or a harder long-horizon task with matched protocols.

---

## Artifact locations

| Content | Where |
|---|---|
| Training run + `latest.ckpt` | private `diffusion_policy` checkout (`data/outputs/…`; match by SHA-256) |
| Primary inference-step ablation tables | copied under [`results/`](results/) |
| Seed-0 timing side study | [`results/ablation_seed0_summary.csv`](results/ablation_seed0_summary.csv) |
| This note | `Week02_Diffusion_Policy/` |

**Checkpoints and the upstream source tree are not copied into this GitHub note** (see [SOURCE_INDEX](results/SOURCE_INDEX.md)).
