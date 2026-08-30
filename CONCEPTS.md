# RoughNet Sentinel-1 Pipeline — Concept Notes

Compiled reference of explanations from working through the Sentinel-1 acquisition,
LiDAR patching, and preprocessing pipeline. Written 2026-08-28.

## Environment & tooling

**micromamba** — a small, fast package/environment manager (a lightweight
alternative to Anaconda/Miniconda). It creates isolated Python environments
with a specific Python version and specific packages, without needing
admin/root access. Used here because the CS workstation's system Python was
3.9 — too old for several required packages (e.g. `click==8.2.1` needs
Python ≥3.10) — and there was no `sudo` access to install a newer Python
system-wide. `micromamba activate roughnet` switches a terminal into the
Python 3.11 environment with all packages installed; a fresh terminal
defaults back to system Python 3.9 unless the activation lines are added to
`.bashrc`.

**Transitive dependencies** — a package doesn't have to be imported directly
by your code to be required. `click` and `click-plugins` are pulled in
because `rasterio` depends on them for its own CLI tooling (the `rio`
command), even though nothing in this project's code calls `click` directly.
This is why `click==8.2.1` showing up in `requirements.txt` isn't a red flag —
it's just what `pip freeze` captured from the environment it was generated on.

## What `01_sentinel1_acquisition.ipynb` actually does

1. **Build a search area (AOI)** from the already-extracted LiDAR patches —
   defines *where* on Earth to look for satellite imagery.
2. **Authenticate with CDSE** (Copernicus Data Space Ecosystem) using `.env`
   credentials to get an access token.
3. **Query the catalogue** for Sentinel-1 scenes matching a date range
   (centered on the LiDAR survey date) and product type.
4. **Select the best matches** — sorts by closeness to the survey date, keeps
   the top `MAX_PRODUCTS` (currently 6).
5. **Download and extract** each selected scene as a `.zip` (SAFE format).

Known gap (left as-is, not fixed): the computed AOI is never actually applied
as a spatial filter in the CDSE query — the query only filters by date and
product type. In practice the download still returned correct, sensible
results, so this wasn't worth fixing for the current experiment.

## Sentinel-1 (radar) vs Sentinel-2 (optical) — why both exist in this project

**Sentinel-2 is optical/passive** — essentially a camera. It measures
sunlight reflected off the ground across visible + near-infrared
wavelengths.
- Only works in daylight.
- Blocked by clouds.
- Its raw data already looks like a recognizable picture (RGB bands are
  genuinely colors).

**Sentinel-1 is radar (SAR) and active** — it emits its own microwave pulses
and measures the echo bounced back.
- Works day or night (brings its own "light").
- Passes through clouds.
- The result has no visible-light meaning — each pixel is backscatter
  intensity (how much energy bounced back), not color. Raw Sentinel-1 images
  look like grainy grayscale speckle, not a photograph.

This is exactly why Sentinel-1 is worth exploring for this Arctic site
(Tuktoyaktuk): it's frequently cloudy/dark for part of the year, conditions
where Sentinel-2 can't see anything usable but Sentinel-1 still works.

## Calibration and georeferencing (what `calibrate_and_warp()` does)

Raw Sentinel-1 pixel values are just uncalibrated digital numbers reflecting
sensor gain settings (which vary across the image, near-range vs far-range) —
not yet physically meaningful.

- **Sigma-nought (σ⁰)** — the standard physical unit for radar backscatter:
  how much radar energy bounced back per square meter of ground, usually in
  dB. It depends on real surface properties:
  - **Roughness** — rough surfaces scatter energy in many directions
    (including back to the sensor, so high σ⁰); smooth surfaces
    (calm water, flat ice) act like a mirror and reflect the beam away (low
    σ⁰). This is the whole physical basis for why Sentinel-1 could plausibly
    predict terrain *roughness* at all.
  - Also depends on moisture/material and incidence angle.
- **Calibration XML → sparse lookup grid** — ESA provides calibration values
  at a coarse set of sample points, not per-pixel.
- **Interpolation** — those sparse values are interpolated to fill in a
  value for every actual pixel at full resolution.
- **Warping via GCPs (Ground Control Points)** — a raw Sentinel-1 image
  isn't on a simple map grid by default. GCPs are points where the exact
  real-world coordinates are known; warping uses them to reproject the image
  so every pixel lands at its correct geographic location.

**What calibration is *for*, precisely:** it makes Sentinel-1 values
internally consistent and physically meaningful (comparable pixel-to-pixel
and scene-to-scene) — a prerequisite for any model to learn a genuine
relationship between backscatter and roughness, rather than learning sensor
noise.

**What calibration is *not* for:** it does not make Sentinel-1 directly
comparable to LiDAR. LiDAR measures elevation (meters); calibrated
Sentinel-1 measures backscatter (dB) — different physical quantities,
different units, never numerically comparable even after calibration.

**What actually connects the two datasets:**
1. *Calibration* → Sentinel-1 values become physically trustworthy on their
   own.
2. *Geometric warping/collocation* (a separate step, `collocate()` in
   notebook 03) → Sentinel-1 and LiDAR pixels are aligned to the same CRS/
   grid, so pixel `(i,j)` in both patches represents the same real-world
   spot.
3. Given aligned, trustworthy pairs, it's the **model** (not calibration)
   that learns the statistical relationship between backscatter patterns and
   roughness patterns during training.

## `build_products()` verification results (8-product tuk set, 2026-08-28)

After expanding the query from 5 to 8 products (fixing the AOI query to use
`convex_hull` instead of a loose bounding envelope, plus a coverage filter —
see the "dead product" fix elsewhere in this doc/changelog), every one of
the 8 outputs (`t0.tif`–`t7.tif`) was individually verified with the
per-band QA check (shape/count/CRS, plus `nonzero_frac`, min/max/mean for
VV and VH):

| Product | Date | Shape | nonzero_frac | Status |
|---|---|---|---|---|
| t0 (`843D`) | 2024-03-18 | 31128×28389 | 0.449 | ✅ |
| t1 (`40A4`) | 2024-03-20 | 31088×27207 | 0.463 | ✅ |
| t2 (`6689`) | 2024-04-01 | 31086×27206 | 0.462 | ✅ |
| t3 (`8A2D`) | 2024-04-13 | 31083×27195 | 0.463 | ✅ |
| t4 (`E833`) | 2024-04-23 | 31133×28398 | 0.449 | ✅ |
| t5 (`25F8`) | 2024-04-25 | 31083×27197 | 0.462 | ✅ |
| t6 (`EE55`) | 2024-05-05 | 31130×28392 | 0.449 | ✅ |
| t7 (`0ED4`) | 2024-05-07 | 31076×27187 | 0.462 | ✅ |

No dead products (all `nonzero_frac` well above zero, none near the ~0
pattern that flagged the earlier `0632`/`4B20` non-contributing products).
VV and VH `nonzero_frac` matched exactly within every product. Three of
the eight (`8A2D`, `E833`, `25F8`) were independently cross-checked against
Michel Tsamados's separately-run Colab notebook, which converged on the
same 3 products via the same convex-hull + coverage-filter logic — full
agreement.

**Interesting pattern**: the 8 products cleanly split into two clusters by
shape and `nonzero_frac`, rather than being scattered continuously:

- **Cluster A** (`t0`, `t4`, `t6`): ~31130×28392, `nonzero_frac` 0.449
- **Cluster B** (`t1`, `t2`, `t3`, `t5`, `t7`): ~31085×27200, `nonzero_frac`
  0.462–0.463

This is consistent with the products coming from (at least) two slightly
different relative orbit tracks — small differences in swath tilt/footprint
between orbit passes shift both the warped bounding-box dimensions and the
fraction of that box actually covered by real data (see the earlier section
on why the bounding box is larger than the real swath). Both clusters are
comfortably healthy; the split itself isn't a defect, just a fingerprint of
which orbit each pass belongs to.

## Using `lidar_patches_tuk_tessa` (Tessa's original patches) instead of your own

Two separate LiDAR patch directories now exist side by side under
`input_data/`:

- `lidar_patches_tuk/` — your own 1708 patches (notebook 02,
  `MIN_VALID_FRACTION = 0.7`, zero-fill convention).
- `lidar_patches_tuk_tessa/` — Tessa's original 1676 patches, obtained via
  Michel's shared Google Drive folder, transferred directly onto the
  workstation (bypassing the Mac entirely, since direct `scp` from a home
  network can't reach the department host without VPN, and Guacamole's
  built-in file transfer was disabled). Flattened after extraction (the
  zip's internal folder caused one level of double-nesting, same class of
  issue as the Sentinel-1 SAFE folders).

**To use Tessa's patches for a `collocate()` run**, no changes are needed to
`build_products()` — it only reads LiDAR_DIR for one file's CRS (see the
earlier section explaining why), and that CRS is the same either way.
Only the `collocate()` step actually depends on which patch directory is
used. To switch:

```python
LIDAR_DIR = REPO_DIR / 'input_data' / 'lidar_patches_tuk_tessa'
OUT_DIR = REPO_DIR / 'input_data' / 's1_patches_tuk_tessa'   # new name — do not reuse s1_patches_tuk
```

Then re-run just the `collocate()` cell (not `build_products()`, which is
unaffected). Using a distinct `OUT_DIR` name keeps this run's output
separate from the existing `s1_patches_tuk/` (built against your own
patches), so both can be compared later rather than one overwriting the
other.

## Adopting Michel's stricter collocation function (`extract_lidar_matched_s1_patches`)

Reading Michel's actual `patching_sentinel1.ipynb` revealed his matching logic
is meaningfully stricter than the original `collocate()` used in this
pipeline:

| | Own `collocate()` | Michel's `extract_lidar_matched_s1_patches` |
|---|---|---|
| Window size | Whatever the reprojected bounds produce (no fixed target) | Fixed target (`s1_patch_size = round(patch_size / resolution)`, e.g. 26 for 256px @ 10m); any window not exactly matching this size is rejected |
| Rejection scope | Skips only the failing product's timestep; patch folder still written with fewer files | Rejects the **entire patch** (all timesteps) if any single product fails the size/bounds check |
| NaN tolerance | Only checks "is everything NaN" (`not np.isfinite(patch).any()`) | Explicit `max_nan_frac` threshold (default 2%) — rejects a patch if *any* product's window has more than a small fraction of NaN pixels, even if not fully empty |

**Option 1 — apply Michel's matcher to your own patches (ready immediately,
no recalibration needed):** Since your own `lidar_patches_tuk` (1708
patches) and your existing calibrated `t0.tif`–`t7.tif` products already
share the same CRS (EPSG:6931), `transform_bounds()` in Michel's function is
a no-op here — no CRS mismatch, so his matcher runs directly against data
you already have. Output goes to a new `s1_patches_tuk_michel/` directory
(kept separate from the existing `s1_patches_tuk/`, which is stale — see
below — and from `s1_patches_tuk_tessa/`).

Expect **fewer than 1708 written patches** — Michel's whole-patch rejection
on any size/bounds/NaN failure is stricter than the original `collocate()`,
so some patches near the true swath edge that the old function would have
partially kept will now be dropped entirely. Fewer patches, but each one
guaranteed clean (exact size, ≤2% NaN) across all 8 timesteps.

**Known stale data note**: at time of writing, `s1_patches_tuk/` (the
original own-patches collocation output) still reflects the *old* 5-product
run (including the two "dead" products `0632`/`4B20`) — it was never
regenerated after the acquisition query was fixed to find 8 good products.
Confirmed via its unchanged `06:58`-this-morning timestamp, predating all
the later `build_products()` work. `s1_patches_tuk_michel/` (Option 1's
output) supersedes it with fresh, correctly-covering data.

**Option 2 — full replication of Michel's methodology for Tessa's patches**:
see the CRS-mismatch section below. Michel's own pipeline calibrates
Sentinel-1 directly into whichever CRS the LiDAR patches he's using are
already in (`dst_crs = ref.crs`, read from the first LiDAR patch file) —
for him, that's Tessa's native EPSG:32608 (UTM Zone 8N), which is *why* his
run never hit the CRS-mismatch/inflation problem documented below. Your own
`build_products()` instead hardcoded `dst_crs = EPSG:6931` (matching your
own patches), which works fine for your own patches but causes the
36×36-instead-of-26×26 inflation when applied to Tessa's differently-CRS'd
patches. To fully replicate Michel's approach for Tessa's patches, a second
calibration pass targeting `dst_crs = EPSG:32608` is required — this is the
deferred "Question 1" item in `QUESTIONS_FOR_MICHEL.md`, planned for after
the baseline training run.

## `s1_patches_tuk_michel` verification results (2026-08-28)

Notebook `06_sentinel1_own_patches_michel_matcher.ipynb` applied Michel's
stricter `extract_lidar_matched_s1_patches` to this project's own LiDAR
patches (`lidar_patches_tuk`, 1708 patches, EPSG:6931) against the
already-calibrated Sentinel-1 products (`t0.tif`-`t7.tif`, also EPSG:6931).

**Match result: 1708/1708 matched, 0 skipped, 0 skipped-NaN.** Every single
patch passed the exact-26x26-size, in-bounds, and <=2% NaN checks for all 8
products with no exceptions. This is the expected outcome, not a
surprising one: the AOI used for the original Sentinel-1 acquisition query
was built directly from these same 1708 patches' own bounding boxes, and
the coverage filter required (and achieved) exactly 100% AOI coverage for
every selected product — so by construction, no patch could fail the
match. The same reasoning as the earlier `s1_patches_tuk_tessa` 100%-match
finding applies here, even more directly since the AOI and the patches are
literally the same source.

**Quantitative collocation-quality check** (LiDAR roughness vs. Sentinel-1
backscatter correlation, `compute_collocation_quality_stats`, computed
across all 1708 patches at t0):

| Band | Correlation: roughness vs. mean (dB) | Correlation: roughness vs. std (dB) |
|---|---|---|
| VV | 0.618 | 0.558 |
| VH | **0.689** | 0.627 |

All four correlations are positive and moderate-to-strong — independent
evidence that the collocation is genuinely spatially aligned (rougher
LiDAR terrain really does correspond to higher/more variable backscatter
in the matched Sentinel-1 window; a misaligned/randomly-paired collocation
would show correlations near zero). **VH correlates more strongly than VV**
in both mean and std, consistent with VH's known sensitivity to structural
complexity (multiple/volume scattering) — physically closer to what
elevation-std "roughness" is actually measuring than VV's general surface
reflectivity sensitivity (see the VV/VH polarization section above).

**Visual check** (`show_multiple_matched_patches`): sampled patches show
full, clean Sentinel-1 speckle texture with no blank/missing regions
across all timesteps, consistent with the 100% match rate above.

**Conclusion:** `s1_patches_tuk_michel` is fully verified — data existence,
per-band validity, and physical plausibility all confirmed. Ready for use
in baseline training without further verification.

## Sentinel-1 vs. Sentinel-2 performance on Tessa's model (2026-08-29)

First trained/evaluated result, using `s1_patches_tuk_michel_provided` (Tessa's
LiDAR patches + Michel's own UTM-8N-calibrated Sentinel-1, 3 timesteps),
compared against Tessa's own reference Sentinel-2 numbers (read from
individual sample rows in her `prediction_evaluation.ipynb` CSV output —
her Pond Inlet/Tuktoyaktuk validation set, not a computed overall mean, so
treat as an indicative range rather than a precise figure):

| Metric | Sentinel-1 (this project) | Sentinel-2 (Tessa's, sample range) |
|---|---|---|
| RMSE (m) | 0.190 | ~0.11-0.135 |
| Bias (m) | -0.0069 | ~-0.001 to -0.002 |
| RMS Height Error (%) | 32.7% | ~3-20% |
| Normal Angle Error (deg) | 1.80 | ~1.5-2.1 |
| JSD | 0.139 | ~0.013-0.096 |
| PSD RMSE | 2.099 | ~0.6-0.86 |
| ZNCC | **0.166** | **~0.74-0.78** |

**Two very different stories depending on the metric:**
- **Magnitude/bias-based metrics** (RMSE, bias, normal angle error): Sentinel-1
  is worse but in the same ballpark — roughly 40-70% higher RMSE, not
  catastrophically different. Normal angle error is essentially comparable.
- **Spatial-pattern-fidelity metrics** (ZNCC, JSD, PSD RMSE): Sentinel-1 is
  dramatically worse. ZNCC in particular (0.166 vs ~0.75) shows the
  predicted spatial pattern barely correlates with ground truth, versus
  strong correlation for Sentinel-2.

**Interpretation, consistent with the qualitative reconstruction grid**: the
Sentinel-1-conditioned model recovers the overall *scale* of terrain
roughness reasonably (why RMSE/bias/angle error aren't wildly off) but
fails to reconstruct the correct *spatial pattern* (why correlation-based
metrics collapse) — consistent with the fine-detail-loss pattern visible in
the qualitative reconstruction grid, and with the resolution-mismatch
hypothesis (Sentinel-1's ~10m native resolution vs. LiDAR's 1m) under test
in `07_sentinel1_coarse_resolution_evaluation.ipynb`.

**Caveats — not a clean, controlled comparison:**
1. **Different experimental design**: Tessa's numbers come from her
   cross-region validation protocol (Pond Inlet + Tuktoyaktuk combined);
   this Sentinel-1 result is a within-Tuk random split on a much smaller
   dataset — the notebook's own closing note already flags this exact
   mismatch.
2. **Her numbers are read from 5 individual sample rows**, not a computed
   mean across her full validation set — an approximate range, not a
   precise benchmark.
3. **Confounded by training scale**: her model was presumably trained on a
   larger, multi-region Sentinel-2 dataset for longer; this Sentinel-1
   model is freshly trained today on ~1676 Tuk-only patches for a limited
   run — this comparison conflates "Sentinel-1 vs Sentinel-2 as a signal"
   with "less training data/time vs. more."

## Two collocation paths, clarified

Two parallel LiDAR-patch/Sentinel-1 pairings exist for this project, both
usable for training against Tessa's model architecture; the choice between
them is about whose LiDAR patches to report against, not a difference in
downstream modeling:

| | LiDAR patches | Sentinel-1 | CRS situation | Status |
|---|---|---|---|---|
| **Path 1** | Own (`lidar_patches_tuk`, 1708, EPSG:6931) | Own DIY-calibrated (EPSG:6931) | Matched — no mismatch | Done — `s1_patches_tuk_michel`, verified above |
| **Path 2** | Tessa's (`lidar_patches_tuk_tessa`, 1676, UTM 8N) | Own DIY-calibrated (EPSG:6931) | Mismatched | Working version exists (`s1_patches_tuk_tessa`, 36x36/~360m-context windows, see CRS-mismatch section); a cleaner UTM-8N-recalibrated version is notebook 05's deferred ~4-5 hour job |

**Confirmed (2026-08-28) via the shared Drive folder**: both pieces of
Michel's own output exist and are accessible:

- `raw_data/tuk_sentinel1_diy_corrected/t0.tif`-`t2.tif` (4.8-5.2GB each) —
  his UTM-8N-calibrated products for `8A2D`, `E833`, `25F8` (3 of the same
  8 products used throughout this project).
- `input_data/s1_patches_tuk/` — his **already-collocated** output: 1676
  patch folders (matching Tessa's exact patch count), each with `t0.tif`,
  `t1.tif`, `t2.tif` (~6KB each) plus a fully-populated `attrs.json`
  (`acquisition_date`, `orbit_direction`, `relative_orbit_number`,
  `polarisation_channels` — real values, unlike this project's own
  `attrs.json` which is currently `null` throughout — see the acquisition-
  date gap noted in the "Adopting Michel's stricter collocation function"
  section above; his file is a concrete template for fixing that).

This means a **zero-computation, CRS-mismatch-free** 3-timestep Path 2
dataset is available for direct download, without running any part of
notebook 05. It can also partially short-circuit the full 8-timestep
version: only the remaining 5 products (`843D`, `40A4`, `6689`, `EE55`,
`0ED4`) would need calibrating to UTM 8N, not all 8.

## Coordinate reference system (EPSG:6931) and why the warped bounding box is larger than the real swath

### What EPSG:6931 is

EPSG codes identify entries in a standardized public registry of coordinate
reference systems (CRSs) — each code maps to one precisely-defined way of
representing locations on Earth as flat X/Y (or lat/lon) coordinates.

**EPSG:6931** is *WGS 84 / NSIDC EASE-Grid 2.0 North*, a **Lambert
Azimuthal Equal-Area** projection centered on the North Pole:

- **Equal-area**: every pixel/cell represents the same physical ground area
  everywhere in the projection (a 10 m pixel is genuinely 10 m × 10 m on the
  ground regardless of where in the scene it sits). This matters for
  roughness/backscatter statistics (RMSE, correlation length, PSD, etc.),
  which would be distorted if pixel area varied by location.
- **EASE-Grid**: developed by NSIDC (the US National Snow and Ice Data
  Center) specifically for polar/cryosphere science — a natural fit for an
  Arctic terrain-roughness project.
- **North**: centered on the North Pole, minimizing distortion specifically
  for high-latitude Arctic sites such as Tuktoyaktuk.

This is different from **EPSG:4326** (plain WGS84 lat/lon), which is *not*
equal-area — a degree of longitude covers a much smaller physical distance
near 69°N than at the equator. EPSG:4326 is adequate for a coarse CDSE
search-area query (notebook 01), but the wrong choice for a pixel grid meant
to align precisely with LiDAR data.

**Why EPSG:6931 specifically in this pipeline:** `build_products()` reads
the CRS directly off a reference LiDAR patch (`dst_crs = ref.crs`) and warps
Sentinel-1 to match it — the LiDAR patches already use EPSG:6931. Sentinel-1
is therefore warped into the same equal-area, 10 m-per-pixel grid as the
LiDAR data, so pixel `(i, j)` in both datasets represents the same
real-world 10 m × 10 m patch of ground — the alignment `collocate()`
depends on.

### Why the warped output is a mostly-empty bounding rectangle

Sentinel-1 flies a near-polar, sun-synchronous orbit whose ground track
crosses the Arctic at an angle relative to true north — so a single IW
swath, viewed on a north-up map, is a **tilted rectangle**, not one aligned
to the map's axes. A GeoTIFF, however, must store data in a plain
axis-aligned grid. `warp_grid()` (via `calculate_default_transform`) has to
compute the smallest axis-aligned box that fully contains the tilted swath,
and that box is necessarily larger than the swath itself:

```
┌─────────────────────────────┐   ← axis-aligned bounding box (stored raster)
│  ░░                          │
│    ░░░                       │
│      ░░░  ← actual tilted    │
│        ░░░   swath data      │
│          ░░░                 │
│            ░░░               │
│              ░░              │
└─────────────────────────────┘
```

The corners outside the tilted swath are filled with zero/background —
this is exactly what shows up as non-real coverage in a valid-pixel-fraction
check.

**Worked example (product `843D`, 2024-03-18, → `t0.tif`):**

| Quantity | Value |
|---|---|
| Source (native satellite geometry) size | 26449 × 16330 ≈ 432 million px |
| Computed output (bounding box) size | 28389 × 31128 ≈ 884 million px |
| Inflation ratio | 884 / 432 ≈ **2.05×** |
| Predicted real-coverage fraction (1 / 2.05) | ≈ 0.49 |
| Measured `nonzero_frac` (VV and VH, both bands) | **0.449** |

The measured fraction is close to the value predicted purely from the
source-vs-output pixel-count ratio, with the small remaining gap explained
by no-data margins already present along the edges of the source SAR
product itself. This agreement is used as evidence that a mid-range
`nonzero_frac` (roughly 0.3–0.5) reflects the expected rotated-swath
geometry rather than a processing defect — as opposed to a `nonzero_frac`
near 0, which is how the two non-contributing ("dead") products in the
5-product test set were identified.

### QA check used for every product

After `build_products()`, each output `t{index}.tif` is checked before
proceeding to `collocate()`:

```python
import rasterio
import numpy as np

for p in products:
    with rasterio.open(p) as src:
        vv = src.read(1)
        vh = src.read(2)
        for name, band in [('VV', vv), ('VH', vh)]:
            valid = np.isfinite(band) & (band != 0)
            # nonzero_frac = valid.mean()
```

Interpretation:
- `nonzero_frac` well above zero and roughly consistent with the
  source/output pixel-count ratio → genuine, correctly-warped data.
- `nonzero_frac` for VV and VH matching each other within one product →
  consistent processing between the two polarizations (they share the same
  acquisition footprint, so a mismatch would indicate a per-band problem).
- `nonzero_frac` near zero → the product contributed no real data (as found
  for two of the original five downloaded products, traced back to an
  overly loose AOI query that let non-overlapping products through; fixed
  by querying with the AOI's convex hull instead of its bounding envelope).

## VV and VH polarizations, and why repeating them to fill 4 channels is valid (with caveats)

Sentinel-1 records two **polarization channels**, not two measurements of the
same thing — each is a genuinely different way of sending/receiving the
radar signal, sensitive to different physical surface properties:

- **VV (Vertical-Vertical)** — transmit vertically, receive vertically
  (co-polarized). Generally the stronger signal; most sensitive to
  surface-level characteristics (roughness, moisture, basic reflectivity).
- **VH (Vertical-Horizontal)** — transmit vertically, receive the portion
  that comes back rotated to horizontal (cross-polarized). Only happens via
  multiple scattering inside a complex 3D structure (vegetation canopies,
  rubble, complex/layered ice). Generally weaker than VV, but sensitive to
  *structural complexity* rather than surface smoothness. VV and VH are
  complementary, not redundant.

**Why they get repeated to fill Tessa's model's 4-channel input slot:**
Tessa's U-Net has a fixed first convolutional layer with weights already
trained assuming exactly 4 input channels (Sentinel-2's R, G, B, NIR). This
is a hard matrix-shape requirement, not a preference — the network simply
cannot run on 2 channels. Something has to map VV/VH into 4 slots. Of the
options (zero-pad the extra slots, invent a synthetic derived band, or
repeat the real data), repetition is the most conservative: it never
discards real signal and never fabricates values, so every channel carries
genuine backscatter measurements rather than blanks or invented numbers.

**The important caveat — this is not a free fix.** Feeding `[VV, VH, VV,
VH]` means channels 1 and 3 are *literally identical*, and so are 2 and 4.
A convolution kernel computes a weighted sum across channels, so for
identical inputs the weights on channels 1 and 3 just add together
(`w1·x + w3·x = (w1+w3)·x`) rather than combining two genuinely distinct
signals the way real R and B channels would be combined. Statistically,
this input looks nothing like real Sentinel-2 data (where all 4 channels
are correlated but distinct), even though it satisfies the architecture's
shape requirement. Repetition is valid in the narrow sense that the math
runs and no information is thrown away — but whether the model's learned
weights, tuned for genuinely-4-distinct-channel statistics, still produce
anything useful when 2 of those channels are duplicates of the other 2, is
exactly the open empirical question this experiment tests, not a guarantee
of success.

## "Picture" vs "numbers" — a common misconception

Both Sentinel-2 and calibrated Sentinel-1 are, at the point they reach the
model, just numeric arrays shaped `(channels, height, width)`. There is no
separate "picture" format a neural network consumes — a digital image *is*
a grid of numbers.

- Sentinel-2: 4 channels (R, G, B, NIR reflectance). Happens to look like a
  photo to a human when rendered, but the network never sees "a photo" — it
  sees 4 channels of floats.
- Calibrated Sentinel-1: channels of backscatter values (VV, VH). Would
  render as grainy speckle to a human, not a recognizable photo, but
  structurally identical input format to the model.

Since the format is identical, Tessa's model's fixed 4-channel input slot
can be fed Sentinel-1 by repeating VV/VH to fill it — architecturally valid.
**What's actually uncertain** is not the format but the *statistics*: the
model learned specific numeric patterns and textures from Sentinel-2
reflectance data (typical ranges, no speckle noise). Radar backscatter has a
very different numeric distribution and texture (notably, speckle — SAR's
characteristic grainy noise, absent in optical imagery). Whether the model's
learned filters generalize to this different-looking numeric input despite
the identical tensor format is the actual open question being tested.

## LiDAR patching: Tessa's original method vs this pipeline's `02_lidar_patch_processing.ipynb`

| Aspect | Tessa's original | This pipeline (as first written) |
|---|---|---|
| Patch size / step | 256px, stride 128 | 256px, step 128 (same) |
| Valid-pixel threshold | **70%** (`valid_threshold = 0.7`) | **2%** (`MIN_VALID_FRACTION = 0.02`) |
| Invalid-pixel fill | Zero-filled | Left as `NaN` |
| CRS handling | LiDAR warped to match the Sentinel-2 grid, then patched | LiDAR patched in native CRS; Sentinel-1 warped to match LiDAR later instead |
| Coupling to imagery | LiDAR + matching S2 patches extracted together in one pass; a LiDAR patch is discarded if S2 doesn't align | LiDAR patches extracted independently; Sentinel-1 pairing happens separately (notebook 03) |
| Region/split metadata | Tags patches with a region ID for spatial train/val splitting | No region tagging; notebook 04 does a plain random shuffle split |

**For the "test Tessa's frozen model on Sentinel-1" experiment**, only the
threshold and fill-convention actually mattered, and were changed to match
her exactly (`MIN_VALID_FRACTION = 0.7`, zero-fill + mask-band convention,
no GeoTIFF `nodata` declared — matching her `LidarS2Dataset` loader's
expectations). The other three differences were deliberately left alone:

- **CRS handling**: no S2 target grid exists for this experiment, and
  notebook 03 already warps Sentinel-1 to match whatever CRS the LiDAR
  patches carry — nothing to gain by replicating the S2-grid warp step.
- **Coupling to imagery**: literally can't be done Tessa's way for
  Sentinel-1 without restructuring the acquisition order (see below) —
  and even then, decoupling doesn't change the final paired output, only
  when Sentinel-1 gets downloaded relative to LiDAR patching.
- **Region tagging**: only matters for train/val splitting during training.
  The stated first experiment is *inference-only* (run Tessa's
  already-trained, frozen model on Sentinel-1 patches to see how it
  performs) — no train/val split happens at all, so region metadata serves
  no purpose here.

**Bit-for-bit identical patches to Tessa's are not achievable regardless**,
because her patch grid depends on warping to a specific Sentinel-2 granule's
exact pixel origin, which we don't have a reference for. This doesn't matter
for a fair comparison, though — the model doesn't know or care about
absolute geographic tile boundaries. What makes a comparison fair is patches
of the same size, same resolution, and the same completeness bar (70% valid
data), which is what was actually matched.

## Why LiDAR patches are extracted before Sentinel-1 is downloaded (and whether that's required)

`aoi_from_lidar_patches()` builds the Sentinel-1 search area by reading the
*already-extracted* LiDAR patches (unioning their bounding boxes) — that's
what creates the "patches must exist first" ordering in this pipeline.

This is a specific implementation choice, not a fundamental requirement. The
AOI could instead be computed directly from the raw LiDAR mosaic's own
bounding box (`box(*src.bounds)`), which only needs the single large
mosaic file — allowing Sentinel-1 to be downloaded independently, or even
before, LiDAR patch extraction, matching Tessa's actual sequencing (acquire
imagery for the study area, then extract/pair patches).

Reordering the *download* step is independent from *how* the two datasets
get paired (one combined extraction pass, Tessa's style, vs. extract LiDAR
fully first and collocate with imagery afterward, this pipeline's style) —
those are two separate decisions. The final paired dataset comes out
identical either way; reordering only changes *when* the download happens,
not *what* data results. Not changed for this project, since the pipeline
already works end-to-end as-is and the AOI isn't even applied as a real
spatial filter in the query yet (see above).

## Preprocessing steps used across the pipeline (2026-08-30)

Full inventory of preprocessing, organized by stage. Useful for a
methodology section, and for spotting where the Sentinel-1 path diverges
from Tessa's original Sentinel-2 path.

### LiDAR (shared by every experiment, notebooks 04/08/09/10/11)

- Read band 1 (elevation/roughness) and band 2 (validity mask) from the
  patch GeoTIFF.
- NaN/inf cleanup (`nan_to_num`).
- **Demeaning**: subtract the patch's own valid-pixel mean before it
  reaches the model. This is on top of Tessa's "RANSAC residual" step,
  which happens further upstream in `src/data/processing.py` (unchanged
  from her baseline) — a plane is fit to the raw elevation and subtracted
  to remove large-scale tilt/slope, leaving a roughness residual. The
  patch-mean subtraction here is a second, local centering on top of that.

### Sentinel-1 (in `LidarS1Dataset.__getitem__`)

Common to all variants:
- Read only the first 2 bands (VV, VH) per timestep.
- NaN/inf cleanup, clamp to a positive floor (`maximum(sar, 1e-12)`) before
  taking a log.
- **Linear power → dB conversion** (`10 * log10`) — standard SAR practice.
- **Bilinear resize** to 256x256 (LiDAR patch size; Sentinel-1's native
  window is ~26x26).

Per-experiment variants:
- **Baseline (04) / real-attrs (09)**: VV/VH **repeated to 4 channels**
  (`[VV, VH, VV, VH]`) — a compatibility adapter for Tessa's model, which
  hardcodes a 4-channel-per-view assumption inside `AttrAwareSpatialPool`
  (see the notebook-08 intro cell for the full explanation).
- **Native 2-channel (08)**: no repetition — real 2 channels per view,
  enabled by a small patch to `ConditionalUNet`/`AttrAwareSpatialPool`
  (`cond_channels_per_view` parameter).
- **Despeckled (10)**: a 5x5 spatial **median filter** applied to each
  band, before the dB conversion (standard order for SAR despeckling —
  filter in linear power domain, then log-transform).

### Metadata / attrs

- **Baseline (04) / native2ch (08) / despeckled (10)**: zero-filled — no
  real per-view metadata reaches `AttrAwareSpatialPool`'s attention
  mechanism at all.
- **Real-attrs (09)**: acquisition age normalized by 30 days (relative to
  the LiDAR survey date), orbit direction encoded as 1.0 (ascending) /
  0.0 (descending), relative orbit number normalized by 175 (the
  Sentinel-1 repeat-orbit cycle length). The remaining attribute slots
  stay zero — Sentinel-1 products don't carry a direct analog to
  Sentinel-2's cloud-cover/sun-angle fields.

### Upstream, before any of the above (notebooks 01-03)

- Sentinel-1 **radiometric calibration** to sigma-nought — either from
  externally-corrected products, or an experimental raw-SAFE
  calibration/GCP-warping path (`WarpedVRT`).
- **CRS reprojection/collocation**, matching Sentinel-1 scenes to LiDAR
  patch extents (this is where the UTM Zone 8N vs. EPSG:6931 mismatch,
  addressed in notebooks 05/06, came from).

### For comparison: Tessa's original Sentinel-2 preprocessing (unchanged)

Per her project README (`tessacannon48_diffusion_ice_mapping.pdf`):
bilinear resize to LiDAR patch size; **global mean/std normalization per
band**, computed across the training set; cloud-cover percentage scaled to
[0,1]; acquisition age as a signed day-offset from the LiDAR survey date;
zenith angle scaled to [0,1]; azimuth angle **sinusoidally encoded** as a
(cos, sine) pair rather than used as a raw angle; training-time random
augmentation.

**Notable gap**: nothing in the Sentinel-1 path has an equivalent of
Sentinel-2's global mean/std normalization step — dB conversion is the
only intensity transform applied to Sentinel-1 anywhere in this pipeline.
Worth flagging explicitly in a methodology write-up rather than leaving it
implicit, since it's an asymmetry between how the two sensors are prepared,
independent of any of the Phase 1 ablations.

## Why despeckling (Phase 1.3) uses a median filter, specifically

SAR imagery has **speckle**: a grainy, salt-and-pepper noise pattern from
coherent-imaging interference between many sub-resolution scatterers within
a pixel. Unlike optical sensor noise (roughly additive, Gaussian-ish),
speckle is **multiplicative** and appears even over physically uniform
terrain — it's an artifact of how radar imaging works, not a property of
the ground. Tessa's architecture was tuned around Sentinel-2's optical
noise characteristics, so it's an open question whether it can see past
SAR's very different noise structure to find the genuine roughness signal
underneath.

**Hypothesis being tested:** the raw, pre-model correlation between
Sentinel-1 backscatter and LiDAR roughness was already measured at
r=0.618-0.689 (`compute_collocation_quality_stats`, notebook 06) —
moderate-to-strong, computed independent of the deep model. If that
correlation is real but speckle is drowning it out before the model can
learn from it, removing speckle first should make the relationship easier
to learn, and ZNCC should improve. If ZNCC doesn't move, that argues
speckle isn't the dominant bottleneck.

**Why a median filter specifically, not a SAR-specific adaptive filter**
(Lee, Frost, Refined Lee — which use local statistics rather than a flat
window): time and simplicity, given a 2-day sprint. A 5x5 spatial median
filter is a one-line, well-understood way to test "does despeckling help
at all" cheaply. If it shows a real effect, that justifies building a more
sophisticated despeckling approach later (Phase 2.2 in
`TODO_next_experiments.md`: a learned, non-local-means-style layer) rather
than investing in it upfront on an unproven hypothesis.

**Why it's applied before dB conversion, not after:** speckle is
multiplicative in the original linear power (sigma-nought) domain — that's
the physically correct place to filter it. Filtering after the log
transform would mean filtering a nonlinearly-distorted version of the
noise.

**Open concern, not yet resolved (asked in `QUESTIONS_FOR_MICHEL.md`,
item 8):** despeckling smooths texture, but texture is arguably part of
the genuine roughness signal, not purely noise. A median filter doesn't
cleanly distinguish "noise being removed" from "real fine-scale roughness
variation being removed" — a null result from this experiment could mean
either "speckle wasn't the bottleneck" or "the filter removed real signal
along with the noise," and this ablation alone can't tell those apart.

## What Phase 1 (notebooks 08/09/10) actually is, in one paragraph

All three notebooks use Tessa's **identical, unmodified architecture** --
same `ConditionalUNet`, same U-Net depth/width, same diffusion scheduler
and sampler, same training loop, same evaluation metric family. The
*only* thing that differs across them is a single preprocessing change
applied to the raw Sentinel-1 data before it reaches that unchanged model:
08 changes channel handling (real 2-channel VV/VH instead of repeated
`[VV,VH,VV,VH]`), 09 changes metadata (real acquisition-age/orbit attrs
instead of zero-filled), 10 changes denoising (a median filter before dB
conversion). Each is a single-variable change against the same baseline
(notebook 04) -- same data, same split, same seed -- so any metric
difference is attributable to that one specific preprocessing choice, not
a confound from several things changing at once.

This design answers one specific question: **is Sentinel-1's weaker
performance (ZNCC 0.166 vs. Sentinel-2's ~0.74-0.78) caused by how the
data is fed into the architecture, or by the architecture itself being
unsuited to SAR?** Phase 1 tests the "how it's fed in" hypothesis first,
since it's cheap and doesn't require touching the model. Phase 2 (a new
SAR-specific layer, e.g. the polarization-ratio feature) only becomes
necessary if none of Phase 1's preprocessing changes meaningfully close
the gap -- at that point the evidence would point at the architecture
itself, not the preprocessing, as the bottleneck. See
`TODO_next_experiments.md` for the full phase breakdown.

## Phase 1 final results summary (2026-08-30)

All four Phase 1 ablations (08 native2ch, 09 realattrs, 10 despeckled) plus
the combined test (12 native2ch+realattrs) are complete, evaluated against
the same 04 baseline and Tessa's Sentinel-2 reference range:

| Experiment | RMSE (m) | Bias (m) | Sigma Err % | Normal Angle (°) | JSD | PSD RMSE | ZNCC |
|---|---|---|---|---|---|---|---|
| 04 baseline (zero-attrs, repeated ch) | 0.190 | -0.0069 | 32.7% | 1.80 | 0.139 | 2.099 | 0.166 |
| 08 native2ch | 0.197 | -0.00004 | 24.74% | 2.05 | 0.099 | 1.688 | 0.192 |
| 09 realattrs | 0.195 | -0.0075 | 29.21% | 1.91 | 0.129 | 1.774 | 0.165 |
| 10 despeckled | 0.21 | -0.01 | 26.78% | 2.00 | 0.11 | 1.61 | **0.14** |
| **12 native2ch+realattrs (combined)** | 0.198 | 0.001 | 24.08% | 2.12 | **0.094** | **1.576** | **0.204** |
| Tessa's Sentinel-2 (sample range) | ~0.11-0.135 | ~-0.001 to -0.002 | ~3-20% | ~1.5-2.1 | ~0.013-0.096 | ~0.6-0.86 | ~0.74-0.78 |

**Individual factor effects on ZNCC:**
- **Native channels (1.1) help on their own**: 0.166 -> 0.192.
- **Real attrs alone don't** (0.166 -> 0.165, flat) — consistent with the
  earlier finding that `AttrAwareSpatialPool`'s attention mechanism has
  little to weight *with* when the channel input is already the
  out-of-distribution repeated `[VV,VH,VV,VH]` (attrs 09 was tested against
  the repeated-channel baseline, not against native2ch).
- **Despeckling alone actively hurts** (0.166 -> **0.14**, and RMSE gets
  worse too: 0.190 -> 0.21, the worst RMSE of all five runs). This is a
  genuine negative result, not a null one.

**Interaction confirmed**: combining native2ch + realattrs (12) reaches
ZNCC=0.204, beating native2ch alone (0.192) by more than realattrs' solo
effect would predict if the two factors were simply additive/independent
(realattrs alone moved ZNCC by ~0, yet adding it on top of native2ch moved
ZNCC by +0.012 beyond native2ch alone). This answers the question raised
mid-sprint ("why wouldn't one uninformative + one informative factor
interact?") — they do interact, mechanistically because both flow through
the same `AttrAwareSpatialPool` computation (`Z = cat([Fv, Amap])`):
real per-view attrs only have a meaningful per-view feature map `Fv` to
attend over once the channel input itself isn't already corrupted by
repetition. 12 is also the best result on JSD and PSD RMSE of all five
Phase 1 runs, and its JSD (0.094) is inside Tessa's Sentinel-2 range.

**Despeckling's negative result, interpreted against Question 8
(`QUESTIONS_FOR_MICHEL.md`)**: the median filter was hypothesized to
either (a) remove noise and reveal a cleaner roughness signal, or (b) not
matter, or (c) remove real signal along with the noise. The result rules
out (a) — despeckling made ZNCC *and* RMSE worse than the un-despeckled
baseline, not just fail to improve them. That's evidence for (c): a 5x5
flat median filter likely smooths away genuine fine-scale backscatter
texture that correlates with roughness, not just multiplicative speckle.
The one metric that did improve, PSD RMSE (2.099 -> 1.61, second-best of
all five), is consistent with this: smoothing removes high-frequency
content indiscriminately, which can move an aggregate frequency-spectrum
metric closer to target even while destroying the pixel-wise spatial
pattern (ZNCC) that depends on that texture being in the right place, and
the flattened predictions even hurt raw magnitude accuracy (RMSE, Sigma
Err %). This is exactly the risk flagged in Question 8's second
sub-question, now with measured evidence behind it rather than a
hypothetical concern — worth raising with Michel as an answered (not just
asked) question.

**Decision, per the rule set in `TODO_next_experiments.md`'s Phase 1
section**: none of the individual ablations (08, 09, 10) closes the ZNCC
gap, and neither does the best combination found (12, ZNCC=0.204 vs.
Tessa's ~0.74-0.78 — still roughly a 3.5-4x gap). Per the pre-committed
decision rule, this is evidence that the bottleneck is not purely
interface-level, and points toward Phase 2 (a genuinely new SAR-specific
architecture layer) or the deferred Planetary Computer RTC path (below) as
the next candidate explanations — contingent on time remaining after the
2-day Phase 1 + Phase 3 sprint and the dissertation-writing deadline.

## Considered-but-deferred: Microsoft Planetary Computer RTC data (2026-08-30)

Planetary Computer's `sentinel-1-rtc` collection provides **radiometric
terrain correction** (DEM-based correction for local incidence angle,
layover, and shadow effects from terrain slope). This is a different axis
of data quality from anything currently being tested in Phase 1:

- **Despeckling (1.3)** tests grainy, multiplicative *noise*.
- **RTC** would fix *geometric/radiometric distortion from terrain slope*
  in the current raw-SAFE calibration path, which does sigma-nought
  calibration and GCP-based warping but explicitly skips terrain
  correction (see `QUESTIONS_FOR_MICHEL.md`, item 6).

Both could plausibly matter independently, but they're not the same
hypothesis and one experiment doesn't stand in for the other.

**Why not added to the current sprint:** switching data sources isn't a
preprocessing tweak like 08/09/10/12 (a code change inside
`LidarS1Dataset`) -- it's a new acquisition pipeline: query Planetary
Computer's STAC API, confirm actual coverage exists for Tuktoyaktuk in
the relevant date range (unconfirmed -- see `QUESTIONS_FOR_MICHEL.md`
item 6), download, then redo collocation/patching against the existing
LiDAR patches. That's closer to redoing notebooks 01-03 for a new source
than a same-day addition, and doesn't fit the 2-day Phase 1 + Phase 3
timeline.

**Decision:** kept as the open supervisor question it already was
(`QUESTIONS_FOR_MICHEL.md` item 6), to be raised as the natural next step
if Phase 1's ablations (native2ch, realattrs, despeckled, combined) don't
meaningfully close the gap -- at that point there'd be a clear, ranked set
of candidate explanations to bring to a supervisor meeting: (a)
architecture doesn't suit SAR (Phase 2), (b) input data's terrain-
correction quality (this Planetary Computer path), (c) something not yet
identified.
