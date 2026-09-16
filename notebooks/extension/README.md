# Extension experiments (added 15–16 Sep 2026)

Notebooks written after the main chain (`pcrtc/`, `ew/`, `dem_unet/`) was complete.
All paths are absolute (`/cs/student/project_msc/2025/aibh/jiayiche/...`), so the
notebooks run from this folder unchanged.

| # | Notebook | Purpose | GPU time | Status | Needed for |
|---|---|---|---|---|---|
| 23 | `23_age_attribute_control` | Age attribute clamped/zeroed on unseen dates + Cambridge Bay | 40 min | **done** — no effect (Δ ≤ 0.002) | private note only, not in the dissertation |
| 24 | `24_standardised_ood_and_samplers` | Part A: standardised checkpoint on unseen dates + Cambridge Bay. Part B: DDIM / PLMS / DDPM on the validation set | A: 3 h (**done**); B: 2 h | Part A done, Part B pending | Part A → Table `tab:std_ood`. Part B → sampler table |
| 26 | `26_standardised_chain` | Re-run every trained configuration with standardised input (7 trainings, resumable) | ~17 h over 2 sessions | **to run** | the master table in Chapter 4 |
| 25 | `25_train_realattrs_ep200` | Real-attribute config at 200 epochs (dB/DDIM) | 5 h | superseded by the `std_realattrs_ep200` entry in 26 | — |
| 28 | `28_despeckle_ablation` | Leakage-free redo of `raw_data/10`: 3x3 and 5x5 median despeckling on the standardised leading configuration, PLMS | ~2.2 h | to run after 26 | one Limitations / Future-work sentence |
| 27 | `27_standardised_figures_and_audit` | PLMS evaluation of the standardised checkpoint with all Chapter-4 figures; preprocessing leakage audit (split overlap + statistics from training ids only) | ~1.5 h | **to run** after 26 | Chapter-4 figures |

## Run order
1. `26` — Run All. When the GPU session ends, Run All again next session; finished configurations are skipped.
2. `24` with `RUN_PART_A = False` — Part B only.
3. `27` — figures and audit for the standardised model (inference only).
4. `pcrtc/20_verdict_recomputation` last cell → paste the eight-column table into the write-up.

## Outputs
All metrics JSONs go to `s1_training_outputs/`; checkpoints to `checkpoints/`.
Standardised-chain files carry the tag `s1_pcrtc_std_<config>_...`.
