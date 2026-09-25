# 03 — Code Map

File/function map for the Week 2 local learning run and inference-step ablation. Paths are relative to a local checkout of [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy). **This GitHub note does not include that source tree.**

---

## 1. End-to-end training path

```
train.py
  → Hydra config (local PushT overrides on official lowdim workspace)
  → TrainDiffusionUnetLowdimWorkspace.run()
       ├── PushTLowdimDataset / DataLoader / LinearNormalizer
       ├── DiffusionUnetLowdimPolicy.compute_loss(batch)
       ├── loss.backward(); optimizer.step(); lr_scheduler.step()
       ├── EMA update (if enabled)
       ├── periodic validation loss
       ├── periodic PushTKeypointsRunner rollout
       └── save checkpoints/latest.ckpt
```

| Stage | Upstream location (local checkout) |
|---|---|
| Entry | `train.py` |
| Workspace | `diffusion_policy/workspace/train_diffusion_unet_lowdim_workspace.py` |
| Dataset | `diffusion_policy/dataset/pusht_dataset.py` |
| Policy | `diffusion_policy/policy/diffusion_unet_lowdim_policy.py` |
| UNet | `diffusion_policy/model/diffusion/conditional_unet1d.py` |
| Task / runner cfg | `diffusion_policy/config/task/pusht_lowdim.yaml` |
| Local training overrides | `diffusion_policy/config/` (Hydra file inheriting official lowdim workspace) |
| Smoke config | same `config/` folder (short debug run) |

---

## 2. Evaluation / reload path

```
eval.py
  → load Workspace checkpoint (dill/pickle)
  → prefer ema_model when present
  → PushTKeypointsRunner.run(policy)
  → eval_log.json + media/*.mp4
```

| Stage | Location |
|---|---|
| Entry | `eval.py` |
| Runner | `diffusion_policy/env_runner/pusht_keypoints_runner.py` |
| Env | `diffusion_policy/env/pusht/pusht_keypoints_env.py` |

`mean_score` in logs = average over rollouts of each episode’s **maximum** reward.

New-process eval artifact (copied): [`results/final_eval.json`](results/final_eval.json).

---

## 3. Inference-step ablation scripts (local only)

```
local ablation runner
  → --trust-checkpoint load latest.ckpt
  → inspect policy dims / N / ema key
  → for each (sampler, K) in randomized order:
        fixed-obs CUDA latency samples
        closed-loop rollouts for each (env_seed, sampling_seed)
        optional video for first pair
  → summary.csv / summary.json / episodes.csv / manifest.json

local plot script
  → read summary.csv
  → score_vs_steps.png, latency_vs_steps.png, speed_quality.png
```

These scripts sit at the **root** of the local `diffusion_policy` checkout (next to `train.py`). They are **not** uploaded into this note (same policy as Week 1 not uploading ACT sources). Behavior and numbers are documented in [06_Inference_Step_Ablation.md](06_Inference_Step_Ablation.md); raw tables are under [`results/`](results/).

---

## 4. Data artifacts

| Artifact | Role |
|---|---|
| `data/pusht/pusht_cchi_v7_replay.zarr` | demonstration dataset (not in this note) |
| `data/outputs/<training_run>/` | formal training run |
| `.../checkpoints/latest.ckpt` | ~1 GB weights — **not** in this note |
| `.../logs.json.txt` | per-batch log; reduced to [`results/epoch_metrics.csv`](results/epoch_metrics.csv) |
| `data/outputs/<ablation_primary>/` | main ablation (5 envs × 3 noise seeds) |
| `data/outputs/<ablation_seed0>/` | seed-0-only side study |

---

## 5. Metric definitions (code-facing)

| Name | Definition in this stack |
|---|---|
| `train_loss` / `val_loss` | noise-prediction MSE |
| `train/mean_score`, `test/mean_score` | mean of episode max rewards over the runner’s init set |
| `policy_median_ms` | median of 20 CUDA-synced `predict_action` calls on one fixed obs |
| `speedup_vs_DDIM_reference` | median(DDIM-100) / median(current DDIM K) |

Do not report mean score as “success rate %.”
