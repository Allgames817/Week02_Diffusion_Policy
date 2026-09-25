# 05 — Behavior Case (env 100000 vs 100003)

Under the Day 6 protocol (same EMA checkpoint, max 300 env steps), aggregate mean scores look high for most DDIM settings. Looking at individual episodes shows a **sampler- and noise-seed-specific** failure that pulls down DDPM and DDIM-20 averages.

Raw per-episode table: [`results/episodes.csv`](results/episodes.csv).  
Videos: [`results/videos/`](results/videos/).

---

## 1. Success reference: env 100000, sampling seed 0

All five conditions reach episode max reward **1.0**. Recorded videos:

| File | Setting |
|---|---|
| [`videos/env100000_noise0/DDPM_100.mp4`](results/videos/env100000_noise0/DDPM_100.mp4) | full-chain DDPM |
| [`videos/env100000_noise0/DDIM_100.mp4`](results/videos/env100000_noise0/DDIM_100.mp4) | DDIM K=100 |
| [`videos/env100000_noise0/DDIM_50.mp4`](results/videos/env100000_noise0/DDIM_50.mp4) | DDIM K=50 |
| [`videos/env100000_noise0/DDIM_20.mp4`](results/videos/env100000_noise0/DDIM_20.mp4) | DDIM K=20 |
| [`videos/env100000_noise0/DDIM_10.mp4`](results/videos/env100000_noise0/DDIM_10.mp4) | DDIM K=10 |

Observable outcome: the gray block covers the green T by the end. DDIM clips are ~126–127 frames; DDPM is longer (~170 frames). Mid-trajectory frames **do not** coincide across samplers even when the final reward matches.

---

## 2. Failure case: env 100003, sampling seed 0

| Setting | Episode max reward | Approx. length | End state (from video) |
|---|---:|---|---|
| DDPM-100 | **0.179** | 300 frames (timeout) | block does **not** cover green target |
| DDIM-20 | **0.179** | 300 frames (timeout) | block does **not** cover green target |
| DDIM-100 | 1.0 | ~159 frames | covers target |
| DDIM-50 | 1.0 | ~162 frames | covers target |
| DDIM-10 | 1.0 | ~127 frames | covers target |

Videos: [`results/videos/env100003_noise0/`](results/videos/env100003_noise0/).

The two failures’ final frames also **do not** match each other. Frame counts are **simulation steps**, not policy latency. Compute wait time is not injected into the physics loop.

---

## 3. Why this matters for the ablation table

From [`results/summary.csv`](results/summary.csv) (5 env seeds × 3 sampling seeds):

| Condition | mean score | descriptive std |
|---|---:|---:|
| DDPM-100 | 0.929 | 0.209 |
| DDIM-20 | 0.935 | 0.210 |
| DDIM-100 | 0.991 | 0.012 |
| DDIM-50 | 0.996 | 0.010 |
| DDIM-10 | 0.987 | 0.034 |

DDPM and DDIM-20 means are dragged by this **one** rollout (0.179). The other two sampling seeds on env 100003 did **not** reproduce that failure in the logged table. Sampling seeds share environments — they are **not** 15 independent environments.

Interpretation:

- High average score can hide systematic or seed-sensitive failure modes (same lesson as Week 1 rollout 33, different mechanism).
- Differences between DDPM-100 and DDIM-20 on this seed are **not** pure “fewer DDIM steps”: DDPM is a different sampler; DDIM-20 also uses a different leading start level than DDIM-100.

---

## 4. What was not claimed

- A full spatial failure map over PushT initial states (no 7×7 grid this week).
- That DDIM-20 is generally worse than DDIM-10 (panel is small; one seed dominates).
- That video length equals wall-clock policy time.
