# pcrtc/06 figures — best result of the study

Destination for the headline plots from `pcrtc/06_train_realattrs.ipynb`
(PC-RTC + real per-view attrs). This is the single most important
experiment in the project — best result on every tracked metric, and the
one that resolves the confusing `04` regression once isolated from `05`.
Git can't track an empty folder, so this file holds it in place until
images are added.

## Requested plots

- **Standard deviation plot** — the GT-vs-pred std scatter from `06`'s
  evaluation cell (`s1_gt_pred_std_scatter.png` on the workstation).
  R²=0.880, the best calibration signal of the whole study.
- **Confidence map** — from `pcrtc/07_uncertainty_confidence_map.ipynb`,
  now pointed at `06`'s checkpoint. **Not generated yet** — that notebook
  got interrupted mid-run and needs to be re-run to completion first (see
  chat history). Copy its output plot in once that finishes.
- **Violin plot** — `06`'s per-metric violin plots across the validation
  set (`s1_pcrtc_metric_violin_plots.png`).
- **Box plot** — see note below; the existing violin plots already
  include quartile markers (`inner='box'`), so a plain box plot of the
  same single-config data would be mostly redundant. A cross-config
  comparison box/violin plot (see "other useful plots") is likely more
  valuable than a single-config box plot here.
- **Circle plot** — the success/failure annotated plot
  (`s1_success_failure_annotated.png`). Note: this was designed for this
  project, not an existing figure from Tessa's dissertation (confirmed by
  checking her PDF directly) — worth describing it that way in any
  write-up rather than attributing it to her.

## Other plots worth including

- **Reconstruction grid** (`s1_patch_reconstructions_pdfs.png`) — S1
  views, GT LiDAR, Pred LiDAR, Error, and per-patch PDF comparison, all in
  one figure. Arguably the single most information-dense qualitative
  figure in the whole project; easy to overlook since it isn't in the
  list above, but worth prioritizing alongside the std scatter.
- **Cross-config comparison chart (03 vs 04 vs 05 vs 06)** — none of the
  existing notebooks produce this, but it doesn't currently exist and
  would be a strong dissertation figure: one bar or box plot per metric
  (RMSE, ZNCC, sigma-error) with all four PC-RTC configs side by side,
  visually showing `06`'s win and the native2ch/realattrs decomposition
  in a single glance, rather than requiring a reader to compare four
  separate tables. I can write the plotting code for this if useful — it
  only needs the four notebooks' saved `*_validation_metrics.json` files,
  no retraining.

## One thing to watch when copying

Several plot filenames are shared across notebooks (not auto-suffixed per
config the way the metrics JSON files are) — running or re-running
another `pcrtc` notebook can silently overwrite `06`'s plots on the
workstation before you copy them out. Grab them now and rename on the way
in (e.g. `pcrtc06_gt_pred_std_scatter.png`) rather than assuming they'll
still be there later.
