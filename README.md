# RoughNet Sentinel-1 Substitution Experiments

Tests whether Sentinel-1 (SAR) can substitute for Sentinel-2 (optical) as
conditioning input to Tessa Cannon's RoughNet diffusion-model pipeline for
LiDAR terrain-roughness prediction, for a dissertation project supervised
by Michel Tsamados. Sentinel-1's appeal is cloud-independence (SAR sees
through cloud cover, unlike optical imagery), at the cost of speckle noise
and a fundamentally different signal type.

## Two experiment tracks

**Track A — raw-SAFE calibration** (`notebooks/01`-`12`): builds a custom
Sentinel-1 calibration pipeline from raw SAFE archives (sigma-nought
calibration, GCP-based georeferencing), then tests preprocessing fixes
(native 2-channel input, real per-view attributes, despeckling, and their
combination) against a zero-attrs/repeated-channel baseline.

**Track B — Planetary Computer RTC data** (`notebooks/pcrtc/01`-`07`):
switches the data source itself to Microsoft Planetary Computer's
`sentinel-1-rtc` collection, which provides DEM-based radiometrically
terrain-corrected backscatter (no custom calibration needed), then repeats
the same preprocessing factorial on this cleaner data source.

Both tracks are matched against the same LiDAR patches and evaluated with
Tessa's original reconstruction metrics (RMSE, ZNCC, JSD, PSD RMSE,
sigma-error, normal-angle-error) for direct comparability.

## Headline result

**Planetary Computer RTC data + real per-view attributes (no channel
repetition removed) is the best result across both tracks:**

| Config | RMSE (m) | ZNCC | JSD | pred_std/gt_std |
|---|---|---|---|---|
| Best raw-SAFE result (native2ch + real attrs) | 0.198 | 0.204 | 0.094 | 0.84 (under) |
| PC-RTC baseline | 0.173 | 0.286 | 0.145 | 0.66 (under) |
| **PC-RTC + real attrs** | **0.159** | **0.519** | **0.054** | **1.02** |
| Tessa's Sentinel-2 (reference) | ~0.11-0.135 | ~0.74-0.78 | ~0.013-0.096 | -- |

Closes the ZNCC gap to Sentinel-2 from ~3.5-4x (best raw-SAFE result) to
~1.4-1.5x, with variance calibration essentially resolved. See
`notebooks/pcrtc/06_train_realattrs.ipynb` for the run, and `CONCEPTS.md`
for the full decomposition (including why the combined native2ch+realattrs
config initially looked like a regression before isolating the two
factors).

## Where to go for more detail

- **`CONCEPTS.md`** -- full methodology, every experiment's results,
  literature comparisons, and reasoning behind each decision. The primary
  running record of this project.
- **`TODO_next_experiments.md`** -- experiment roadmap and status (what's
  done, what's next, decision rules for when to move between phases).
- **`QUESTIONS_FOR_MICHEL.md`** -- open questions for the supervision team,
  with results appended as they land.
- **`PHASE2_ARCHITECTURE_CANDIDATES.md`** -- ranked candidates for a
  potential new architecture layer (DEM conditioning, distribution-matching
  loss, and others), with a time-boxed recommendation.

## Setup

See `requirements.txt` for dependencies and `.env.example` for required
environment variables (CDSE credentials, optional Planetary Computer
subscription key). Data paths in the notebooks point to a CS department
shared workstation (`/cs/student/project_msc/2025/aibh/jiayiche/`) and will
need adjusting for a different environment.
