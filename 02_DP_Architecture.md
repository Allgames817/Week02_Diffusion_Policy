# 02 — Diffusion Policy Architecture (lowdim)

Maps the Week 2 mental model to the official `DiffusionUnetLowdimPolicy` used in the local learning run. Primary upstream files (not vendored here): `diffusion_policy/policy/diffusion_unet_lowdim_policy.py`, `model/diffusion/conditional_unet1d.py`, `workspace/train_diffusion_unet_lowdim_workspace.py`.

---

## 1. High-level diagram

```
Training
────────
  batch obs [B, 16, 20]     # dataset window
  batch act [B, 16, 2]

  obs_cond = obs[:, :2, :]  # n_obs_steps = 2
           → flatten → global_cond [B, 40]

  sample t ~ Uniform{0..N-1},  ε ~ N(0,I)
  noisy_act = √ᾱ_t · act + √(1-ᾱ_t) · ε
  pred_ε = CondUnet1D(noisy_act, t, global_cond)
  loss = MSE(pred_ε, ε)

Inference (eval / ablation)
───────────────────────────
  obs_cond → global_cond [B, 40]
  x ~ N(0,I)  shape [B, 16, 2]
  for k = 1..K:
      pred_ε = CondUnet1D(x, t_k, global_cond)
      x ← DDPM or DDIM step(x, pred_ε, t_k)
  action_pred = unnormalize(x)
  execute action_pred[:, 1:9]   # n_action_steps = 8, To-aligned
```

---

## 2. Fixed hyperparameters (this Week 2 setup)

| Parameter | Value | Notes |
|---|---|---|
| Observation dim | 20 | PushT keypoints (`pusht_lowdim`) |
| Action dim | 2 | agent xy |
| `horizon` ($T_p$) | 16 | |
| `n_obs_steps` ($T_o$) | 2 | condition uses only these steps |
| `n_action_steps` ($T_a$) | 8 | executed slice length |
| UNet widths | `[256, 512, 1024]` | official |
| `num_train_timesteps` ($N$) | 100 | |
| Beta schedule | `squaredcos_cap_v2` | from checkpoint config |
| Prediction type | epsilon | noise prediction |
| EMA | on | eval loads `ema_model` |
| Batch size | 32 | local training config |
| Epochs | 100 | counters: epoch 99, global_step 33599 |

Local Hydra overrides live under `diffusion_policy/config/` in the private checkout and inherit `train_diffusion_unet_lowdim_workspace`.

---

## 3. Conditioning

The dataset may return a full observation window of length 16, but `compute_loss` / `predict_action` condition only on the first $T_o{=}2$ steps:

$$
\text{global\_cond} = \mathrm{flatten}(o_{t-1:t}) \in \mathbb{R}^{40}
$$

Future observations inside the training window are **not** fed to the network as conditions. Do not confuse “dataset tensor length 16” with “model sees 16 observation steps.”

---

## 4. Temporal alignment and action slice

After denoising, the policy returns a length-16 action trajectory in normalized space, then unnormalizes. Execution uses a fixed slice consistent with observation alignment. In this configuration the executed block is:

```text
action_pred[:, 1:9]   # length Ta = 8
```

So each policy call plans $T_p{=}16$ steps but commits $T_a{=}8$ environment actions before the next replan (subject to the runner’s control loop).

---

## 5. Schedulers: DDPM vs DDIM

| | Training | Native inference | Inference-step ablation |
|---|---|---|---|
| Noise levels | $N{=}100$ | often $K{=}N$ DDPM | DDIM $K\in\{100,50,20,10\}$ + full DDPM ref. |
| Retrain needed? | — | — | **No** for DDIM $K$ (same $\beta$ array) |
| $\eta$ | — | — | DDIM $\eta{=}0$ |

**Leading spacing** (diffusers 0.11.1 DDIM): fewer steps skip levels; the **first** noise index also changes (e.g. $K{=}10$ starts at 90, not 99). Reducing $K$ is therefore “fewer updates **and** a different starting level,” not only an early `break` inside a 100-step loop.

This environment pins **diffusers 0.11.1**. Sparse DDPM timesteps are unsafe in that version’s step/variance indexing; the ablation therefore keeps **full-chain DDPM** as reference and uses **DDIM** for the pure-$K$ sweep.

---

## 6. EMA vs ACT temporal aggregation

| Mechanism | What is averaged | When |
|---|---|---|
| Diffusion Policy EMA | network parameters | during training |
| ACT temporal aggregation | overlapping predicted actions | at inference |

Both improve robustness in their papers/codebases, but they are **not** interchangeable concepts.

---

## 7. Training vs inference summary

| | Training | Inference |
|---|---|---|
| Actions | clean + noised | sampled from noise |
| Loss | MSE on noise | none |
| Conditioner | `global_cond` from $T_o$ obs | same |
| Steps | one random $t$ per sample | $K$ sequential updates |
| Weights | online `model` + EMA update | typically `ema_model` |

The ablation manifest confirms eval used `state_dict_key: ema_model`.
