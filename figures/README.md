# Figures

Destination for the headline result plots, copied over from the
workstation's `s1_training_outputs/` directory. Git can't track an empty
folder, so this file also holds it in place until images are added.

## What's worth copying in first

Priority: **`pcrtc/06`'s outputs** (the best result of the entire study —
PC-RTC + real attrs). From that notebook's run:
- The GT-vs-pred standard deviation scatter (R²=0.880 calibration plot).
- The patch reconstruction grid + per-patch PDF comparison.
- The metric violin plots (RMSE, ZNCC, JSD, etc. across the validation set).
- The success/failure annotated circle plot.

Worth adding for comparison, if useful:
- The equivalent plots from `pcrtc/03` (baseline) and the best raw-SAFE
  result (`raw_data/12`), to show the progression across the project.

## One thing to watch when copying

Several of these plots are saved under **the same filename regardless of
which notebook produced them** (e.g. `s1_gt_pred_std_scatter.png`,
`s1_patch_reconstructions_pdfs.png`, `s1_pcrtc_metric_violin_plots.png`) —
they aren't automatically suffixed per config the way the metrics JSON
files are. Running a later notebook (or re-running an earlier one) can
silently overwrite an earlier plot on the workstation before you get a
chance to copy it out. Worth grabbing `06`'s plots now and renaming them
on the way in (e.g. `pcrtc06_gt_pred_std_scatter.png`) rather than
assuming they'll still be there later.
