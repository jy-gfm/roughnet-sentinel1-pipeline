# Extension experiments (added 15–16 Sep 2026)

Notebooks written after the main chain (`pcrtc/`, `ew/`, `dem_unet/`) was complete.
All paths are absolute (`/cs/student/project_msc/2025/aibh/jiayiche/...`), so the
notebooks run from this folder unchanged.

| # | Notebook | Purpose | GPU time | Status | Needed for |
|---|---|---|---|---|---|
| 23 | `23_age_attribute_control` | Age attribute clamped/zeroed on unseen dates + Cambridge Bay | 40 min | **done** — no effect (Δ ≤ 0.002) | private note only, not in the dissertation |
| 24 | `24_standardised_ood_and_samplers` | Part A: standardised checkpoint on unseen dates + Cambridge Bay. Part B: DDIM / PLMS / DDPM on the validation set | A: 3 h; B: 2 h | **done** | Part A → Table `tab:std_ood`. Part B → sampler table |
| 26 | `26_standardised_chain` | Re-run every trained configuration with standardised input (8 trainings incl. 200 epochs, resumable) | ~10 h | **done** | the master table in Chapter 4 |
| 25 | `25_train_realattrs_ep200` | Real-attribute config at 200 epochs (dB/DDIM) | 5 h | superseded by the `std_realattrs_ep200` entry in 26 | — |
| 28 | `28_despeckle_ablation` | Leakage-free redo of `raw_data/10`: 3x3 and 5x5 median despeckling on the standardised leading configuration, PLMS | ~2.2 h | **done** — 0.296 / 0.277 vs 0.370 | Extensions rows + one Discussion paragraph |
| 27 | `27_standardised_figures_and_audit` | PLMS evaluation of the standardised checkpoint with all Chapter-4 figures; preprocessing leakage audit (split overlap + statistics from training ids only); Section 8: confidence deciles + confident-quartile figure | ~1.5 h | **done** (Section 8 pending) | Chapter-4 figures |
| 29 | `29_valbuffer_more_training_data` | Same validation set (255), training set enlarged by buffering only at the validation boundary (>= 128 m); leading configuration, seeds 42 and 43 | ~2.2 h per seed | **to run** | data-limitation test; one row pair in `tab:extensions` |

## Run order (remaining)
1. `27` Section 8 (last two cells) — confidence deciles and figure.
2. `29` — Run All (resumable; seed 43 can be skipped if time is short).
3. `30_reference_split` — zone 13 (reconstructed from the reference's x-binning rule; no `region.json` on the workstation) held out, no buffer; leading configuration; ~1.5 h.

Not run, and why: Pond Inlet (no Sentinel-1 within a year of the survey, HH-only);
Tuk + Cambridge Bay joint training (Cambridge Bay radar is 402-409 days after its LiDAR,
so the pairs show different ice -- the unseen-date test shows such pairs carry no placement signal).

## Outputs
All metrics JSONs go to `s1_training_outputs/`; checkpoints to `checkpoints/`.
Standardised-chain files carry the tag `s1_pcrtc_std_<config>_...`.
