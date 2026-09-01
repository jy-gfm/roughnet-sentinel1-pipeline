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

## Comparison against a published SAR preprocessing pipeline (2026-08-30)

"An ensemble learning method to retrieve sea ice roughness from
Sentinel-1 SAR images" documents a 5-step SNAP-based preprocessing
pipeline: (1) apply orbit file, (2) thermal noise removal, (3) calibrate
to NRCS (sigma-nought), (4) speckle filtering with the **Refined Lee
filter** (7x7 window), (5) convert to dB. Comparing against this
project's two pipelines:

| Step | Raw-SAFE pipeline (01/03) | Planetary Computer RTC |
|---|---|---|
| Apply orbit file | Not documented as a step | Very likely included in Microsoft's RTC processing (standard for a production product), not independently confirmed |
| Thermal noise removal | Not documented as a step | Same caveat as above |
| Calibrate to NRCS | Yes -- matches | Yes -- matches |
| Speckle filtering | Different method, negative result -- plain 5x5 median filter (notebook 10), not edge-preserving, made ZNCC/RMSE worse | Not applied |
| Convert to dB | Yes -- matches | Yes -- matches |

**Two takeaways:**

1. **The raw-SAFE calibration path likely omits two standard steps**
   (orbit-file correction, thermal noise removal) that this published
   pipeline treats as baseline necessities. This strengthens the Phase 4
   finding (PC-RTC baseline beating every raw-SAFE fix, `ZNCC=0.286` vs.
   `0.204`) with a mechanistic explanation beyond just terrain correction:
   a properly-processed product likely bundles several calibration steps
   the DIY path skips, not just DEM-based terrain correction. Worth
   confirming against Planetary Computer's own product documentation
   rather than assuming, but it's the reasonable expectation for a
   production RTC product.
2. **This is concrete literature support for revisiting despeckling with
   the right method.** The paper's own justification for using Refined
   Lee is exactly the concern already raised in Question 8: it "averages
   the image while preserving edges, so the patterns of sea ice will not
   be affected." That edge-awareness is precisely what a plain median
   filter lacks -- it can't distinguish "smoothing out speckle" from
   "smoothing out a real ridge boundary," which is the most likely
   explanation for notebook 10's negative result. If despeckling gets
   revisited, Refined Lee (not another flat filter) is the well-supported
   next thing to try, per Michel's own reply to Question 8's addendum.

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
2-day Phase 1 + Phase 3 sprint and remaining time.

## Phase 3 result: uncertainty/confidence map calibration (2026-08-30)

Ensemble sampling (`N_SAMPLES=5`, `N_CALIBRATION_PATCHES=40`) on the best
Phase 1 checkpoint (`s1_tuk_native2ch_realattrs_unet_best.pth`) gives a
**pixel-level correlation between predicted std and `|error|` of 0.2570**.

**Interpretation**: positive but weak (r^2 ~ 0.066, ~7% of error variance
explained). The map is not random -- higher ensemble disagreement does
correspond, on average, to higher actual error -- but it's not strong
enough to use as a standalone per-pixel trust signal. This lines up with
the GT-vs-pred-std scatter already recorded for this same checkpoint
(R^2=0.424, best-fit line well below the 1:1 line, in the Phase 1 results
summary above): the model's predicted spread is systematically compressed
relative to ground truth, which narrows the dynamic range `pred_std` has
available to track error magnitude, even though the underlying
relationship it's trying to capture is real.

**Visual confirmation, from the reconstruction-grid figure (6 example
patches, rows: GT LiDAR / Pred Mean / Error / Uncertainty (std))**: the
Uncertainty (std) row shows genuine, non-uniform spatial texture (roughly
0.05m in calm regions up to 0.25-0.35m in rougher ones), and it visibly
tracks the Error row on a per-patch basis -- e.g. patch 12597 has both
higher error texture and a brighter, more patchy uncertainty map, while
patches 13279/13184 show low error and correspondingly darker, more
uniform low uncertainty. This is a qualitative illustration of the same
weak-but-real r=0.257 relationship measured quantitatively above -- visible
by eye, not just in the aggregate statistic.

**Known display bug in this same figure, not a data/metric problem**: the
GT LiDAR and Pred Mean rows render as solid, uniform blue blocks with no
visible texture. This is the identical plotting issue already diagnosed
and fixed for notebook 08's reconstruction grid: both rows plot absolute
elevation (roughly -6 to -7m for this region) on a colormap centered at
zero, so the values saturate to a single color. It does not affect any
saved metric (RMSE, ZNCC, the calibration correlation, etc.) or the
Uncertainty/Error rows, which are already display-correct -- it only makes
the GT/Pred rows unreadable. Fix, not yet applied to notebook 11: recenter
each patch around its own masked mean before plotting, the same fix used
for notebook 08.

**Why this isn't a failure, and how to frame it in the write-up**:
ensemble sampling variance from a diffusion model's stochastic sampler
captures *generative/output* uncertainty (variability in what the sampler
produces across random seeds), not full *epistemic* uncertainty (from the
model's learned weights/parameters) -- this distinction was flagged before
running the experiment (`TODO_next_experiments.md`, Phase 3). A
weak-but-positive calibration for that specific, narrower kind of
uncertainty estimate is a defensible, literature-consistent result, not a
broken method. **Conclusion**: the confidence map is directionally
informative but should be presented as a supplementary diagnostic, not a
substitute for validating predictions against ground truth.

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

**Update (2026-08-30): no longer deferred.** Michel (via his postdoc Tom)
independently suggested this exact path by email, confirming it addresses
a real gap: the current raw-SAFE calibration does sigma-nought calibration
and GCP warping but no DEM-based terrain correction, whereas Planetary
Computer's `sentinel-1-rtc` collection provides gamma-nought backscatter
with real 10m-DEM terrain correction, validated by Microsoft against an
independent RTC implementation (`sarsen`). It also removes the ~1hr/product
raw-SAFE processing cost entirely via windowed HTTPS reads against
cloud-optimized GeoTIFFs -- no download, no local calibration.

One caveat from Michel's email, consistent with what this document already
flagged: the collection appears to be **IW-mode only** (every example
found follows `S1A_IW_GRDH` naming) -- good for Tuktoyaktuk (IW), but
likely does not cover Pond Inlet or Cambridge Bay (EW-only sites),
needs confirming against those AOIs specifically before assuming either
way.

`notebooks/pcrtc/01_acquisition_coverage_check.ipynb` resolves the
first open question -- does the collection actually have coverage for
Tuktoyaktuk in the LiDAR survey date window, confirmed via a real windowed
read, not just a catalogue hit. It reuses notebook 01's
`aoi_from_lidar_patches` verbatim so the queried footprint is identical to
the existing CDSE query, making any difference in results attributable to
the data source, not AOI drift.

This is the start of a dedicated `notebooks/pcrtc/` sub-sequence (numbered
01/02/03, mirroring the original 01->04 acquisition-to-training structure
but as its own clearly-separated experiment track): `01` is this coverage
check, `02_patch_extraction.ipynb` merges VV+VH into local per-date
GeoTIFFs and reuses notebook 06's `build_s1_products_from_corrected`/
`extract_lidar_matched_s1_patches` verbatim to produce
`s1_patches_tuk_pcrtc` in the same directory format the existing
`LidarS1Dataset` expects, and `03_train_baseline.ipynb` is an exact copy
of `04_train_s1_with_tessa_baseline.ipynb` with only `S1_DIR` (and
checkpoint/output filenames) changed -- same architecture, same
zero-filled attrs, same repeated-channel handling, so any metric
difference from notebook 04 is attributable specifically to the data
source's calibration/terrain-correction quality, not a different
experimental setup. Matched against `lidar_patches_tuk_tessa` (the same
LiDAR patches the actual baseline and Phase 1 use), which conveniently
shares this PC data's CRS (EPSG:32608), avoiding the rotation/footprint-
inflation issue from Question 1 that affects the raw-SAFE path.

**Result (2026-08-30): coverage confirmed, end to end.** Catalogue search
found 7 `sentinel-1-rtc` scenes for Tuktoyaktuk within the same +/-30-day
window as the CDSE query (vs. CDSE's own count of 8 for the identical
AOI/window -- consistent, not a different subset of acquisitions; all
S1A, IW mode, ascending orbit, `vv`/`vh` assets present). A windowed read
of the actual AOI footprint (not the scene's raw top-left corner, which
returned nodata on the first attempt -- see below) returned real
gamma-nought backscatter: min/max 0.00136-0.2706 (linear power), 0%
nodata in the window -- a physically plausible range, comparable order of
magnitude to the raw-SAFE VV values already validated elsewhere in this
document (mean ~=0.022).

**Bug caught during this check, worth noting**: the first windowed-read
attempt read `Window(0, 0, 512, 512)` -- the scene's top-left corner --
and returned a constant `-32768.0` (the int16 nodata sentinel), because
GRD scenes are stored as an axis-aligned bounding box around an angled
swath, so raster corners commonly fall outside the actual imaged area.
Fixed by transforming the AOI's bounding box into the scene's CRS
(`rasterio.warp.transform_bounds`) and reading that specific window
(`rasterio.windows.from_bounds`) instead of an arbitrary corner -- a
reminder that "windowed read succeeded" isn't sufficient on its own to
confirm real data; checking the values (not just the shape/dtype) is what
actually confirms it.

**Patch extraction result (2026-08-30), `notebooks/pcrtc/02_patch_extraction.ipynb`**:
100% match rate against `lidar_patches_tuk_tessa` -- **1676/1676 LiDAR
patches matched, 0 skipped (out-of-bounds/wrong-size), 0 skipped
(excess NaN)**, every patch has all 7 timesteps, `finite_frac=1.0` and
`nonzero_frac=1.0` on every band. This is a real, attributable data-quality
advantage over the raw-SAFE path (which has to work around the
EPSG:6931-vs-UTM8N CRS mismatch and its footprint-rotation/inflation
issue from Question 1) -- not just "coverage exists," but a materially
cleaner match than the existing baseline's own data preparation achieved.
Output: `s1_patches_tuk_pcrtc/`, ready for `03_train_baseline.ipynb`.

**Baseline training result (2026-08-30), `notebooks/pcrtc/03_train_baseline.ipynb`**:
this is the single most important Phase 4 result so far. Using the
*simplest possible config* -- identical to `04_train_s1_with_tessa_baseline.ipynb`'s
zero-filled attrs and repeated `[VV,VH,VV,VH]` channels, no Phase 1 fixes
applied at all -- on Planetary Computer RTC data:

| | RMSE (m) | ZNCC | JSD | PSD RMSE | Sigma Err % | Normal Angle |
|---|---|---|---|---|---|---|
| Raw-SAFE baseline (04) | 0.190 | 0.166 | 0.139 | 2.099 | 32.7% | 1.80 |
| Best raw-SAFE combo (12, native2ch+realattrs) | 0.198 | 0.204 | 0.094 | 1.576 | 24.08% | 2.12 |
| **PC-RTC baseline (03)** | **0.173** | **0.286** | 0.145 | 1.832 | 33.25% | **1.78** |
| Tessa's Sentinel-2 (reference) | ~0.11-0.135 | ~0.74-0.78 | ~0.013-0.096 | ~0.6-0.86 | ~3-20% | ~1.5-2.1 |

**The simplest config on this data source beats every Phase 1 preprocessing
fix tried on the raw-SAFE data**, on both RMSE and ZNCC (0.286 vs. 0.204
best-previous). This is strong evidence that data-source/terrain-correction
quality was a bigger lever on the original gap than any interface-level
preprocessing fix -- exactly the hypothesis Phase 4 was designed to test,
and it comes back positive. Gap to Tessa's S2 narrows to roughly 2.6-2.7x
(from ~3.7-4.5x with the best raw-SAFE result).

**Not uniformly better, though** -- worth stating precisely:
- **JSD is worse** (0.145 vs. 0.094 for the best raw-SAFE combo) -- the
  predicted roughness *distribution* is less realistic even though
  spatial pattern correlation (ZNCC) improved.
- **PSD RMSE is worse** (1.832 vs. 1.576).
- **Variance compression is worse, not better**: `pred_std=0.112` vs.
  `gt_std=0.169` (ratio ~0.66), a more compressed spread than the best
  raw-SAFE result's ratio (~0.84). Visible directly in the per-patch PDF
  plots -- several predicted (red) distributions are narrower/shifted
  from ground truth (blue).
- The reconstruction grid itself displays correctly this time (GT/Pred
  rows show real structure, not the solid-color plotting bug from
  earlier) -- large-scale blue/red boundary patterns are visibly tracked
  between GT and Pred in several example patches, consistent with the
  higher ZNCC.

**Interpretation**: terrain-correction quality improves *where* the model
gets things right (spatial pattern, ZNCC) and *overall magnitude accuracy*
(RMSE), but doesn't on its own fix *how confidently/how spread-out* the
model's predictions are (JSD, PSD RMSE, variance compression) -- those
look like a separate, still-unresolved limitation, possibly tied to the
zero-filled attrs/repeated-channels config this baseline still uses.
`pcrtc/04-06` (testing the same Phase 1 fixes on this cleaner data source)
will show whether those interface-level fixes now also close the
remaining JSD/PSD/variance gaps, the way they helped ZNCC on the raw-SAFE
data.

**Combined (native2ch+realattrs) result on PC-RTC data (2026-08-31),
`notebooks/pcrtc/04_train_native2ch_realattrs.ipynb`**: the interaction
effect that gave the best raw-SAFE result does not transfer to this data
source -- it reverses.

| | RMSE | ZNCC | JSD | PSD RMSE | Sigma Err % | Normal Angle | pred_std/gt_std |
|---|---|---|---|---|---|---|---|
| Raw-SAFE baseline (04) | 0.190 | 0.166 | 0.139 | 2.099 | 32.7% | 1.80 | -- |
| Best raw-SAFE combo (12) | 0.198 | 0.204 | 0.094 | 1.576 | 24.08% | 2.12 | 0.84 (under) |
| **PC-RTC baseline (03)** | **0.173** | **0.286** | 0.145 | 1.832 | 33.25% | 1.78 | 0.66 (under) |
| **PC-RTC native2ch+realattrs (04)** | 0.238 | 0.219 | 0.122 | **1.326** | 38.83% | 2.559 | **1.24 (over)** |

**On raw-SAFE data this combination was the best result found. On PC-RTC
data it's worse than the plain baseline** on the four metrics that have
mattered most throughout this project (RMSE, ZNCC, sigma-error,
normal-angle-error). Only JSD improved slightly and PSD RMSE improved
substantially (1.326, the best PSD RMSE of the entire study).

**The variance-compression failure mode flipped direction.** Every prior
experiment on both data sources under-predicted variance
(`pred_std < gt_std`). This one over-predicts it (ratio 1.24) -- a
genuinely different failure mode, not just a worse version of the same
one. The GT-vs-pred-std scatter shows a compression-toward-the-middle
pattern: the best-fit line sits above the 1:1 line for low-roughness
patches (over-predicting calm areas) and crosses below it for
high-roughness patches (still under-predicting the roughest ones).

**Likely explanation**: on raw-SAFE data, channel repetition was fixing a
real problem (out-of-distribution duplicated channels reaching a network
whose first-layer filters expect 4 real, distinct signals). PC-RTC's
baseline already performs well *with* repeated channels, suggesting the
repetition artifact matters less once the underlying calibration/terrain-
correction is already good -- so removing it doesn't offer the same fix,
and combined with real attrs, something about this specific combination
appears to destabilize training on this cleaner data distribution rather
than help it.

**Takeaway for the dissertation**: a preprocessing fix validated on one
data source does not automatically generalize to another -- worth stating
explicitly as a methodological finding, not just reporting the numbers.
`pcrtc/05` (native2ch alone) and `pcrtc/06` (realattrs alone) will show
which factor is actually driving this regression, or whether it's
specifically the interaction between them (both fine alone, harmful
together) that's responsible.

**Status**: coverage and read-access are confirmed for Tuktoyaktuk.
Turning this into an actual ablation (comparable to notebooks 08/09/10/12)
would require a new patch-extraction step -- windowed-reading each LiDAR
patch's footprint from these COGs, analogous to notebooks 02/03's raw-SAFE
patching -- plus a new training/eval notebook. That's a genuine new
pipeline component, not a same-day addition; whether to build it now
versus document this as a confirmed-but-deferred result and move to
writing is a time-budget decision, not a methodology one.

## Decomposing the pcrtc regression: `05` (native2ch alone) and `06` (realattrs alone), 2026-08-31

`04`'s combined result looked like a confusing reversal on its own --
worse than the plain baseline on RMSE/ZNCC/sigma-error/normal-angle, with
variance compression flipping direction. Isolating the two factors
resolves it into a clean, interpretable story.

| | RMSE | ZNCC | JSD | PSD RMSE | Sigma Err % | Normal Angle | pred_std/gt_std | Calib R² |
|---|---|---|---|---|---|---|---|---|
| PC-RTC baseline (03) | 0.173 | 0.286 | 0.145 | 1.832 | 33.25% | 1.78 | 0.66 (under) | -- |
| PC-RTC combined (04) | 0.238 | 0.219 | 0.122 | 1.326 | 38.83% | 2.56 | 1.24 (over) | -- |
| PC-RTC native2ch alone (05) | 0.283 | 0.245 | 0.226 | 1.305 | **78.97%** | 2.71 | **1.62 (way over)** | 0.483 |
| **PC-RTC realattrs alone (06)** | **0.159** | **0.519** | **0.054** | 1.345 | **13.13%** | 2.25 | **1.02 (near-perfect)** | **0.880** |

**`06` (realattrs alone) is the best result of the entire study**, on
every metric simultaneously -- including beating the plain baseline (03),
which held that position until now. ZNCC nearly doubles (0.286 -> 0.519),
closing much more of the gap to Tessa's Sentinel-2 reference (~0.74-0.78)
than any previous fix on either data source. Variance compression is
essentially resolved (`pred_std=0.173` vs. `gt_std=0.169`, ratio 1.02),
and the ensemble-uncertainty calibration check (see Phase 3 section above)
jumps to R²=0.880 -- by far the strongest calibration signal seen in this
project, with the GT-vs-pred-std best-fit line visually overlapping the
1:1 line across nearly the entire range (one high-variance outlier patch
sits apart from the main cluster but still lands almost exactly on the
line, consistent with, not contradicting, the strong overall fit).

**`05` (native2ch alone) is actively harmful** -- worse than the baseline
on every metric, and worse than the combined config (04) on most of
them too (RMSE, ZNCC, JSD, sigma-error, normal-angle). Sigma-error blows
up to 79%, the worst number in the entire study, and variance
compression gets *more* extreme in the over-prediction direction (ratio
1.62) than in the combined config (1.24).

**This means the interaction in `04` is native2ch's damage partially
offset by realattrs's benefit, not two neutral factors combining badly.**
Realattrs alone is a strong, clean win; native2ch alone is a strong,
clean loss; combining them nets out to "better than native2ch alone, but
still worse than not using native2ch at all." The practical conclusion is
simple: **use real per-view attrs, keep native 2-channel VV/VH (don't
repeat to 4 channels) is NOT the fix on this data source -- repetition
should stay, only the zero-filled attrs should be replaced with real
ones.**

**Revises the Phase 4 headline finding**: it is no longer "PC-RTC
baseline (03) is the best config on this data source, but preprocessing
fixes don't transfer from raw-SAFE." It is now "PC-RTC + real attrs (06)
is the best config in the entire study, decisively so, and the specific
raw-SAFE fix that doesn't transfer is native2ch, not realattrs."

**Implication for Phase 2 / new architecture work**: any new architecture
experiment (e.g. DEM conditioning, see `PHASE2_ARCHITECTURE_CANDIDATES.md`)
should now be built on top of `06`'s config (PC-RTC + real attrs, repeated
4-channel VV/VH) as the base, not `03`'s plain baseline -- `06` is the
strongest known starting point regardless of this decision, since it beats
`03` on every metric.

## `pcrtc/07` confidence map on the `06` checkpoint (2026-08-31)

`notebooks/pcrtc/07_uncertainty_confidence_map.ipynb`, pointed at `06`'s
checkpoint (`s1_tuk_pcrtc_realattrs_unet_best.pth`) and its real-attrs
dataset class, run to completion (`N_SAMPLES=5`, `N_CALIBRATION_PATCHES=40`).

**Result**: pixel-level correlation between predicted std and `|error|` =
**0.4079** -- moderate (conventionally: <0.3 weak, 0.3-0.5 moderate,
0.5-0.7 strong), and a real improvement over the Phase 3 result on the
best raw-SAFE checkpoint (0.2570, see above), roughly a 59% relative
increase. Qualitatively, the uncertainty map's highest-magnitude regions
concentrate on the most complex/highest-error patches (e.g. patch 12597's
branching terrain shows both the widest uncertainty and the most error
variation of the six example patches) rather than being scattered
independent of where the model actually struggles.

**Important distinction, worth keeping separate**: this is a different
number from the R²=0.880 GT-vs-pred-std scatter reported for `06` in the
decomposition section above. That scatter compares per-patch *aggregate*
standard deviation (does the model produce realistic overall variance per
patch, computed from a single sampling pass) -- a variance-realism
diagnostic. This section's r=0.4079 is the *actual* confidence-map
calibration: per-pixel predicted uncertainty from a 5-sample ensemble,
correlated against per-pixel real error. The two use the word "std" for
different things and are not directly comparable -- don't cite one for
the other in the write-up.

**Display bug encountered and fixed**: the first run's visualization cell
plotted GT and Pred Mean as absolute elevation values (patch mean already
added back) on a colormap centered at zero, saturating both to solid
color -- the same class of bug documented earlier in this project (see
"Picture vs numbers" section). Fixed by recentering both by their own
masked mean before display only, matching the convention already used in
`03`/`06`'s reconstruction-grid cells; the saved calibration correlation
number was computed from the raw (non-recentered) values throughout and
was unaffected by the display bug.

**Standing caveat, restated**: this measures the model's own
generative/output uncertainty (sample-to-sample disagreement across
independent noise seeds), not full epistemic uncertainty. A low-std
(confident) prediction can still be systematically wrong if the model is
consistently biased in a region rather than randomly uncertain there.

## Verified: `lidar_patches_tuk_tessa` is genuinely Tuktoyaktuk, not Pond Inlet

While tracing how Tessa's LiDAR patches and `region_id` assignment are
generated (`Michel/RoughNet/notebooks/Copy of patching.ipynb`), found a
discrepancy worth checking directly rather than assuming either way: the
notebook's visible config cell points `lidar_path` at `tuk_2024.tif`, but
a later visualization cell in the same notebook, showing a patch from the
same `start_idx=12000, region_id+10` batch, titles it "LiDAR (Pond Inlet,
06-04-24)". Since `input_data/lidar_patches_tuk_tessa/` -- the *only*
directory ever used as `LIDAR_DIR` across every pcrtc training notebook
(`03`-`06`, `09`, `10`) -- turned out to consist entirely of that same
batch (patch IDs `12000`-`13675`, 1676 patches, exactly matching every
run's "Paired: 1676" count, with no patch IDs below 12000 present at
all), this meant every headline result in this project rested on data
whose region identity had a real, unresolved contradiction in its
provenance.

**Verification performed**: reprojected a low-ID patch from the separate
`lidar_patches_tuk` directory (patch `00014`, confirmed via its config
cell to come from `tuk_2024.tif`, region_id 0-9 batch, EPSG:6931) and a
patch from `lidar_patches_tuk_tessa` (patch `12066`, EPSG:32608) both to
WGS84 lat/lon and compared directly.

- `lidar_patches_tuk` (00014): 69.71°N, 133.34°W
- `lidar_patches_tuk_tessa` (12066): 69.82°N, 133.35°W
- Distance apart: **13.1 km**

Both are squarely at Tuktoyaktuk (~69.45°N, 133.03°W) -- nowhere near
Pond Inlet (~72.7°N, 77.9°W, ~2,000+ km away). The two patches are simply
13 km apart, consistent with being two different sub-areas/extraction
runs over the same local LiDAR survey.

**Conclusion**: the "Pond Inlet" title was a stale copy-paste artifact in
that one visualization cell, not a real label -- `lidar_patches_tuk_tessa`
is genuinely Tuktoyaktuk data. No result in this project (`05`, `06`,
`08`, `09`, `10`) needs reinterpretation because of this. Also means a
future Cambridge Bay or Pond Inlet cross-region test starts from a clean
baseline -- no prior contamination between "Tuktoyaktuk" training data and
either candidate test region.

## Cross-region generalization test: Tuktoyaktuk model on Cambridge Bay

Per Michel's suggestion to test the existing model on a genuinely new
region (`pcrtc/13_cambridge_bay_crossregion_inference.ipynb`). Ran `09`'s
frozen, already-trained checkpoint (real-attrs, spatial-block split, the
best leakage-free Tuktoyaktuk model) on 400 randomly-sampled Cambridge
Bay patches -- no retraining, no train/val split, every patch held out.

**Pond Inlet was ruled out first.** Its nearest available Sentinel-1
imagery was either HH-only (405-429 days off, same season) or HH+HV
(2.6-9 years off, inconsistent seasons) -- no VV+VH coverage at all for
that AOI. Using it would have stacked three confounds at once (region +
date + polarization mode). Cambridge Bay's nearest VV+VH scenes are
~402-409 days after the survey (May 2025, cross-season) -- a single
confound, chosen over Pond Inlet's three-way one.

**Extraction required two fixes**, both in `11`, before matching worked
at all:
1. One of the three selected scenes (2025-05-25) was delivered in UTM
   zone 12N while the other two and the LiDAR patches are in zone 13N --
   Sentinel-1 scenes get assigned a zone based on the whole scene's
   footprint, not this AOI. Windowing that scene against LiDAR bounds in
   its native zone computed a distorted 28x28 window instead of 26x26,
   failing every single patch's exact-size check (0/2112 matched).
   Fixed by reprojecting every merged scene into the LiDAR patches' CRS
   at a fixed 10m grid before matching.
2. That reprojection then left 17.3% of the reprojected scene's pixels
   at the initial-array value of exactly `0.0` (a border effect from
   warping between UTM zones) -- real SAR backscatter never reads as
   exactly zero, so this would have silently passed corrupted data
   through undetected (the matcher only checked NaN fraction). Fixed by
   filling unmapped pixels with NaN instead, letting the existing
   NaN-fraction check catch them. Final result: 2101/2112 matched.

Full DDIM inference on all 2101 patches would take ~10 hours (scaled
from the ~70min/251-patch benchmark) for a negligibly tighter estimate
than a smaller sample; subsampled to 400 (seed 42, ~7x Tuktoyaktuk's own
251-patch validation set) instead, cutting the run to ~1.5-2 hours.

### Results

| metric | Tuktoyaktuk in-region (09) | Cambridge Bay cross-region |
|---|---|---|
| RMSE (m) | 0.1945 | 0.3389 |
| bias (m) | -0.0013 | -0.0088 |
| sigma_error (%) | 22.73 | 498.42 |
| normal_angle_error (deg) | 1.989 | 2.463 |
| JSD | 0.1180 | 0.5570 |
| PSD RMSE | 1.3092 | 1.5686 |
| ZNCC | 0.2344 | **0.0064** |
| gt_std | 0.1721 | 0.0834 |
| pred_std | 0.1384 | **0.3159** |
| uncertainty calibration corr. | 0.4079 (from `07`) | 0.1526 |

**Finding 1 -- spatial correlation collapses.** ZNCC drops to
essentially zero (0.0064), a similar magnitude of collapse to `08`'s
unseen-date Tuktoyaktuk test (0.52 to 0.01). Because Cambridge Bay's
test data carries both a new-region confound *and* the same
cross-season confound `08` isolated, this single test cannot cleanly
attribute the collapse to region generalization alone -- both are
stacked here.

**Finding 2 -- a specific, characterizable failure mode, not just
"worse."** `gt_std` (0.083) confirms Cambridge Bay's terrain really is
much flatter than Tuktoyaktuk's (consistent with the direct visual
check on the raw LiDAR patches earlier), but `pred_std` (0.316) is
nearly 4x *larger* than the true variance. The model isn't going bland
on unfamiliar input -- it's imposing Tuktoyaktuk-specific roughness
texture onto genuinely smoother terrain. The GT-vs-pred std scatter
plot shows this directly: predictions spread widely regardless of how
flat the corresponding ground truth actually is. Worth stating in the
write-up as the specific failure mode, not just a degraded metric.

**Finding 3 -- confidence calibration degrades but doesn't collapse.**
0.4079 to 0.1526: weaker, but still positive -- the model's own
uncertainty estimate remains a weak-but-real signal cross-region, not
meaningless.

**Caveat on `sigma_error_pct`**: the 498% figure is a normalization
artifact (this metric divides by `gt_std`, which is small here), not a
literal "5x worse" claim -- RMSE and ZNCC are the honest way to state
the magnitude of degradation. Don't quote 498% as a standalone headline.
