# Source Index

Maps every number in this note to a local artifact. Checkpoint identity is the **SHA-256** below; folder names on disk may differ by machine.

Upstream code: [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy). 
Week 1 companion: [Allgames817/Week01_ACT](https://github.com/Allgames817/Week01_ACT).

---

## 1. What is **not** in this GitHub repo

| Item | Reason |
|---|---|
| `latest.ckpt` (~1.0 GB) | exceeds GitHub file limit; keep local |
| PushT `.zarr` dataset | large; download from official data site |
| Full `diffusion_policy` source tree | note-only repo (same as Week 1 vs ACT) |
| Local ablation / plot scripts | live in local checkout root |
| Raw per-batch training log | reduced to `epoch_metrics.csv` |
| Fixed timing observation tensor | timing helper only |

---

## 2. Checkpoint (local)

| Field | Value |
|---|---|
| Role | Self-trained PushT lowdim EMA policy |
| SHA-256 | `922e11a246a4484e7d59956b9fa36da88cbfdf8d2406c937f92357b2cd09001d` |
| Eval weights | `ema_model` |
| Counters | epoch 99, global_step 33599 |
| Local path | under `data/outputs/` in the private `diffusion_policy` checkout (match by SHA-256) |

---

## 3. Copied results in this note

| File | Source |
|---|---|
| [`epoch_metrics.csv`](epoch_metrics.csv) | last row per epoch from training `logs.json.txt` |
| [`final_eval.json`](final_eval.json) | new-process `eval.py` log |
| [`summary.csv`](summary.csv) / [`summary.json`](summary.json) | primary inference-step ablation (5 envs × 3 sampling seeds) |
| [`episodes.csv`](episodes.csv) | same primary ablation |
| [`latency_samples.csv`](latency_samples.csv) | same primary ablation |
| [`manifest.json`](manifest.json) | run metadata (paths sanitized for publication) |
| [`ablation_seed0_summary.csv`](ablation_seed0_summary.csv) | seed-0-only timing side study |
| [`ablation_seed0_summary.json`](ablation_seed0_summary.json) | same |
| [`comparison_table.md`](comparison_table.md) | Chinese ablation table copy |
| [`CONCLUSION_zh.md`](CONCLUSION_zh.md) | Chinese ablation conclusion copy |
| [`videos/`](videos/) | curated rollout clips |

---

## 4. Figures

| Figure | How produced |
|---|---|
| [`figures/train_val_loss.png`](../figures/train_val_loss.png) | plotted from `epoch_metrics.csv` |
| [`figures/rollout_score.png`](../figures/rollout_score.png) | plotted from `epoch_metrics.csv` |
| [`figures/score_vs_steps.png`](../figures/score_vs_steps.png) | from primary ablation `summary.csv` |
| [`figures/latency_vs_steps.png`](../figures/latency_vs_steps.png) | same |
| [`figures/speed_quality.png`](../figures/speed_quality.png) | same (English labels; y-axis zoomed) |

---

## 5. Local directories (author machine — do not move)

Exact folder names stay on the private checkout so experiments remain findable. They are **not** required to read this note.

| Content | Role |
|---|---|
| PushT lowdim training + `latest.ckpt` | training run (seed 42, 100 epochs) |
| Primary DDIM/DDPM ablation outputs | 5 env seeds × 3 sampling seeds |
| Seed-0 timing side study | latency fluctuation check |
| Curated videos / Chinese table drafts | packaging helpers |
| This note | `Week02_Diffusion_Policy/` |
