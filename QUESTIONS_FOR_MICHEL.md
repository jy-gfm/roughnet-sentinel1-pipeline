# Questions for Michel — Sentinel-1 Pipeline

Compiled 2026-08-28, based on work completing the Sentinel-1 acquisition,
calibration, and LiDAR collocation pipeline for Tuktoyaktuk.

**Note on sequencing:** Question 1 (CRS mismatch) will be revisited after
the baseline training run completes — not blocking current progress.

## 1. LiDAR patch CRS mismatch — Tessa's patches vs. my own

Tessa's original LiDAR patches (`lidar_patches_tuk`, shared via your Drive
folder, 1676 patches) are stored in **EPSG:32608 (UTM Zone 8N)**. My own
patches (`lidar_patches_tuk`, 1708 patches, generated from the raw mosaic)
are in **EPSG:6931** (the polar EASE-Grid used for the Sentinel-1
calibration output).

`collocate()` reprojects each LiDAR patch's bounding box into the
Sentinel-1 CRS before cropping. Since my own patches already share the same
CRS as the Sentinel-1 data, this is a no-op — the collocated crop is
exactly 256m × 256m, matching the LiDAR patch precisely.

For Tessa's patches, the reprojection from UTM 8N to EPSG:6931 is a real
transform, and because Tuktoyaktuk sits at high latitude, the two grids'
"north" directions differ by close to 45° at this location. The
axis-aligned bounding box needed to contain the reprojected (now rotated)
256m square inflates to ~360m × 360m (a factor of ≈√2, consistent with a
~45° rotation). So every Sentinel-1 crop collocated against Tessa's patches
carries ~360m of context instead of exactly 256m — the corners of the crop
extend beyond the true LiDAR footprint.

**Question:** Is this acceptable (extra spatial context, no data
corruption), or would you prefer `collocate()` be modified to warp/crop
precisely to the true rotated 256m footprint when using Tessa's patches?

## 2. Which LiDAR patch set to use as the actual experiment input

I now have both patch sets fully collocated with Sentinel-1
(`s1_patches_tuk` against my own 1708 patches, `s1_patches_tuk_tessa`
against Tessa's 1676 patches). Given the CRS issue above only affects the
Tessa-patch version:

**Question:** Should I use my own patches (exact footprint, but not
Tessa's original patches) or Tessa's patches (original/authoritative, but
with the footprint caveat above) for the actual experiment — or run both
and report whether it makes a measurable difference?

## 3. Temporal window — 8 products (±30 days) vs. 3 products (±14 days)

My Sentinel-1 query (AOI convex hull + ≥90% coverage filter) found 8
usable IW products within ±30 days of the LiDAR survey date. Restricting
to ±14 days (matching the original Sentinel-2 pipeline's window) leaves
only 3 of those 8 (`8A2D`, `E833`, `25F8`) — the same 3 your own
diagnostic notebook found independently.

**Question:** For a fair comparison against the Sentinel-2 baseline, should
I restrict to the same ±14-day window (3 products), or is it acceptable —
maybe even a point in Sentinel-1's favor — to use the full 8, since SAR's
cloud-independence is part of what's being tested?

## 4. VV/VH channel repetition to fill the model's 4-channel input

Tessa's `ConditionalUNet` has a fixed first-layer input of 4 channels
(Sentinel-2's R/G/B/NIR). Sentinel-1 only provides 2 channels (VV, VH). My
current plan is to repeat `[VV, VH, VV, VH]` to satisfy the architecture's
shape requirement (see `CONCEPTS.md` for the detailed reasoning and
caveats already written up).

**Question:** Is channel repetition an acceptable adaptation for this
experiment, or would you recommend a different approach (e.g. modifying
the first conv layer to accept 2 channels and reinitializing those
weights, or zero-padding instead of repeating)?

## 5. Training approach — fresh training vs. inference on your trained model

**Question:** Given the missing `.pth` checkpoint issue (Tessa's repo has
no `models/` folder committed), can you confirm training a fresh model
with Tessa's architecture (Option A) is the right path, rather than waiting
on Tessa's checkpoint for inference-only testing (Option B)? You'd
mentioned leaning this way already — just want to confirm before
committing significant compute time to a full training run.

## 6. Raw SAFE calibration — is the current approach sufficient?

My Sentinel-1 calibration/warping pipeline (`build_products()`) does
sigma-nought radiometric calibration and GCP-based georeferencing, but
does **not** include full terrain correction (no DEM-based correction, no
local-incidence-angle correction, no layover/shadow correction) — it's
explicitly the "experimental raw-SAFE path" mentioned in the project
documentation, as opposed to using externally pre-corrected GeoTIFFs.

**Question:** Is this level of correction sufficient for the scope of this
dissertation, or should I pursue proper terrain-corrected products before
proceeding to training?

You mentioned Microsoft's Planetary Computer — its `sentinel-1-rtc`
collection provides Sentinel-1 backscatter that's already been
radiometrically terrain-corrected (DEM-based), which would solve this gap
without me needing to implement terrain correction myself. Note:
`pystac-client` and `planetary-computer` are already in `requirements.txt`
and `.env.example` has a placeholder for `PC_SDK_SUBSCRIPTION_KEY` — this
was apparently anticipated before this pipeline went down the CDSE/raw-SAFE
route instead. If time allows, I'd like to try both: keep the current
raw-SAFE pipeline as one path, and pull the same date range/AOI from
Planetary Computer's RTC collection as a second, terrain-corrected path,
and compare. Two things to confirm before I start that: (1) does the RTC
collection actually have coverage for Tuktoyaktuk in this date range, and
(2) is "try both and compare" the right scope, or would you rather I just
switch entirely to Planetary Computer?

**Update, 2026-08-30**: Tom's email confirmed this is worth trying and
flagged that the collection looks IW-mode only (fine for Tuktoyaktuk, not
for Pond Inlet/Cambridge Bay). Ran
`notebooks/pcrtc/01_acquisition_coverage_check.ipynb` to check (1)
above -- **confirmed**: 7 scenes found for Tuktoyaktuk in the same
+/-30-day window as my existing CDSE query (vs. CDSE's 8 for the identical
AOI), all IW/ascending with `vv`/`vh` assets, and a windowed read of the
actual AOI returned real gamma-nought backscatter (0.0014-0.27, 0%
nodata) -- so coverage and data access both check out for Tuktoyaktuk.
Still deciding whether to build the full patch-extraction/retraining
comparison given the time left, or document this as confirmed-viable and
treat it as future work in the write-up -- your read on which is the
better use of remaining time would help.

## 7. Scope — Tuktoyaktuk only, or extend to Pond Inlet / Cambridge Bay?

Your diagnostic notebook found **zero** `IW_GRDH_1S` products for Pond
Inlet or Cambridge Bay within ±60 days — only `EW_GRDM_1S` (coarser, ~40m
resolution, and possibly different polarization — HH/HV rather than
VV/VH, still to be confirmed).

**Question:** Is the dissertation scope meant to stay Tuktoyaktuk-only, or
should I plan to extend to the other two regions using EW-mode data
(accepting the coarser resolution and possible polarization difference)?

## 8. Despeckling with a median filter — appropriate method, or risks removing real signal?

As part of the Phase 1 ablation series (`TODO_next_experiments.md`), I'm
testing whether SAR speckle noise (the multiplicative, coherent-imaging
noise pattern inherent to radar, absent in optical imagery) is obscuring
the roughness-backscatter relationship the model is trying to learn. My
current implementation applies a simple 5x5 spatial median filter to each
VV/VH band, in the linear power domain, before dB conversion — chosen for
simplicity/speed under a tight timeline, not because it's the standard
SAR-specific despeckling method (vs. e.g. Lee, Frost, or Refined Lee
filters, which use local statistics adaptively rather than a flat window).

**Question 1:** Is a plain median filter a reasonable first test, or is it
crude enough that a null result (no ZNCC improvement) would be
uninformative — i.e. should I use a proper adaptive SAR despeckling filter
instead before concluding anything either way?

**Question 2 (the one I think matters more):** Despeckling smooths
texture — but texture is arguably part of the genuine roughness signal
I'm trying to predict, not purely noise. Is there a real risk that a
median filter removes some of the actual backscatter-roughness
relationship along with the speckle, rather than cleanly separating
"noise" from "signal"? If so, is there a principled way to tell, for this
specific dataset, how much of the fine-scale VV/VH texture is speckle
versus genuine sub-patch roughness variation?

**Result, added 2026-08-30**: I ran this ablation. Despeckling made things
*worse*, not just unhelpfully neutral — ZNCC dropped from 0.166 (baseline)
to 0.14, and RMSE got worse too (0.190 -> 0.21, the worst RMSE across all
my Phase 1 ablations). The one metric that improved was PSD RMSE
(2.099 -> 1.61). My reading is that this is evidence for the risk in
Question 2 above: the filter likely smoothed away genuine fine-scale
roughness texture along with speckle, hurting both spatial pattern
fidelity (ZNCC) and magnitude accuracy (RMSE), while the aggregate
frequency-spectrum metric improved simply because smoothing removes
high-frequency content indiscriminately. Keeping Question 1 open since I
only tested one crude filter — I'd still like your view on whether a
proper adaptive despeckling method (Lee/Frost/Refined Lee) might behave
differently, or whether this result is answer enough not to pursue it
further given the time left.

**Literature note, added 2026-08-30**: found direct support for this in
"An ensemble learning method to retrieve sea ice roughness from
Sentinel-1 SAR images" — they use the Refined Lee filter specifically
because it "averages the image while preserving edges, so the patterns
of sea ice will not be affected." That's exactly the property my flat
median filter lacks, and the most likely explanation for why it hurt
rather than helped. If you think it's worth one more try given the time
left, Refined Lee (not another flat filter) is the well-supported next
step.

## Terminology reference

**Sigma-nought (σ⁰)**: the standard physical unit for radar backscatter —
a linear power ratio describing how much radar energy bounces back per
unit ground area. Depends on surface roughness (rough surfaces scatter in
many directions, including back to the sensor, so high σ⁰; smooth surfaces
like calm water or flat ice act like a mirror and reflect the beam away,
so low σ⁰), moisture, material, and incidence angle. This is the physical
quantity my calibration pipeline outputs, in **linear** units (e.g. my
`t0.tif` VV band has mean ≈ 0.022).

**Decibels (dB)**: a logarithmic representation of sigma-nought, computed
as `10 * log10(sigma0)`. SAR data is conventionally displayed/reported in
dB (including in your own plots, which use a -16 to -26 dB colorbar)
because backscatter values span a huge dynamic range — a log scale
compresses very bright targets and typical terrain into a more
manageable, comparable range. My pipeline currently stores values in
linear sigma-nought, not dB; converting between the two for display or
comparison is a simple one-line transform, not a reprocessing step (e.g.
my linear mean of 0.022 ≈ -17 dB, consistent with the range your plots
showed).
