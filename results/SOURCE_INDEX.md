# Source Index

Maps every number in this note to a local artifact. **Do not move or overwrite** the training/ablation directories when maintaining the note.

Upstream code: [real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy).  
Week 1 companion: [Allgames817/Week01_ACT](https://github.com/Allgames817/Week01_ACT).

---

## 1. What is **not** in this GitHub repo

| Item | Reason |
|---|---|
| `latest.ckpt` (~1.0 GB) | exceeds GitHub file limit; keep local |
| PushT `.zarr` dataset | large; download from official data site |
| Full `diffusion_policy` source tree | note-only repo (same as Week 1 vs ACT) |
| `day6_ablation.py` / `day6_plot.py` | live in local checkout root |
| Raw `logs.json.txt` (3.4 MB, per-batch) | reduced to `epoch_metrics.csv` |
| `fixed_observation.pt` | timing helper only |

---

## 2. Checkpoint (local)

| Field | Value |
|---|---|
| Relative path | `data/outputs/day5_pusht_20260925_003955_seed42/checkpoints/latest.ckpt` |
| SHA-256 | `922e11a246a4484e7d59956b9fa36da88cbfdf8d2406c937f92357b2cd09001d` |
| Origin | self-trained Day 5 run |
| Eval weights | `ema_model` |
| Counters | epoch 99, global_step 33599 |

---

## 3. Copied results in this note

| File | Source |
|---|---|
| [`epoch_metrics.csv`](epoch_metrics.csv) | extracted from Day 5 `logs.json.txt` (last row per epoch) |
| [`day5_final_eval.json`](day5_final_eval.json) | Day 5 `final_eval/eval_log.json` |
| [`summary.csv`](summary.csv) / [`summary.json`](summary.json) | `day6_repeats_20260925_154951/` (**primary** table) |
| [`episodes.csv`](episodes.csv) | same repeats run |
| [`latency_samples.csv`](latency_samples.csv) | same repeats run |
| [`manifest.json`](manifest.json) | same; checkpoint path sanitized to repo-relative |
| [`ablation_seed0_summary.csv`](ablation_seed0_summary.csv) | `day6_ablation_20260925_152122/` (**side study**, seed 0 only) |
| [`ablation_seed0_summary.json`](ablation_seed0_summary.json) | same |
| [`comparison_table.md`](comparison_table.md) | Chinese Day 6 table copy |
| [`CONCLUSION_zh.md`](CONCLUSION_zh.md) | Chinese Day 6 conclusion copy |
| [`videos/`](videos/) | curated clips from `day6_record/videos/` |

---

## 4. Figures

| Figure | How produced |
|---|---|
| [`figures/train_val_loss.png`](../figures/train_val_loss.png) | plotted from `epoch_metrics.csv` |
| [`figures/rollout_score.png`](../figures/rollout_score.png) | plotted from `epoch_metrics.csv` |
| [`figures/score_vs_steps.png`](../figures/score_vs_steps.png) | Day 6 `day6_plot.py` on repeats `summary.csv` |
| [`figures/latency_vs_steps.png`](../figures/latency_vs_steps.png) | same |
| [`figures/speed_quality.png`](../figures/speed_quality.png) | `day6_record/speed_quality.png` (y-axis 0–1) |

---

## 5. Local directories (do not move)

| Content | Path in local `diffusion_policy` checkout |
|---|---|
| Day 5 formal train | `data/outputs/day5_pusht_20260925_003955_seed42/` |
| Day 6 primary ablation | `data/outputs/day6_repeats_20260925_154951/` |
| Day 6 seed-0 side study | `data/outputs/day6_ablation_20260925_152122/` |
| Curated Day 6 record | `data/outputs/day6_record/` |
| This note | `Week02_Diffusion_Policy/` |
