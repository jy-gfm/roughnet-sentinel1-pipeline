# Sentinel-1 Roadmap: Ablations -> New Architecture -> Uncertainty

Context: `s1_patches_tuk_michel_provided` evaluation gave ZNCC=0.166 vs
Tessa's Sentinel-2 reference of ~0.74-0.78 -- comparable RMSE/bias, but much
weaker spatial correlation. This roadmap has three phases, in order:

1. **Phase 1** -- test whether interface-level differences between how
   Sentinel-1 and Sentinel-2 are fed into Tessa's *unmodified* architecture
   explain the gap (channel repetition, zero-filled attrs, speckle noise).
2. **Phase 2** -- if Phase 1 doesn't close the gap, build a genuinely new
   model based on Tessa's architecture, adding SAR-specific layers (e.g. a
   polarization-ratio feature) rather than just changing what's fed into
   the existing layers.
3. **Phase 3** -- since diffusion models are inherently probabilistic,
   generate a per-pixel confidence/uncertainty map from sampling variance,
   and check whether it's calibrated against real terrain error before
   deciding it's a usable result.

See `CONCEPTS.md` for full background and the S1-vs-S2 comparison table.

---

## Phase 1: Interface-level ablations

### 1.0. Coarse-resolution evaluation -- DONE (2026-08-29)

Result over 251 validation patches (pooled 10x10 -> ~26x26, matching S1
native resolution):

| | Full-res (1m) | Coarse (~10m) |
|---|---|---|
| RMSE (m) | 0.188 | 0.165 |
| ZNCC | 0.183 | 0.212 |
| JSD | 0.136 | 0.245 |

ZNCC only moved 0.183 -> 0.212 -- a small bump, not the "substantially
higher" outcome that would have pointed to resolution mismatch as the main
cause. **Conclusion: the weak spatial correlation is not primarily a
resolution-limit artifact.** This rules toward experiments 1.1-1.3 below
(model-side bottlenecks: channel repetition, zero-filled attrs, speckle)
rather than "S1 is just coarser so of course it looks worse." Saved to
`/cs/student/project_msc/2025/aibh/jiayiche/s1_training_outputs/s1_coarse_vs_fine_comparison.json`.

### 1.1. Native 2-channel architecture (no channel repetition) -- IN PROGRESS

Tests whether feeding `[VV, VH, VV, VH]` (out-of-distribution for filters
trained on real 4-channel R/G/B/NIR) is itself degrading performance,
independent of whether the underlying signal is informative.

Implemented in `notebooks/08_sentinel1_native_2channel_ablation.ipynb`.
Required a backward-compatible patch to `tessa_baseline/src/model/unet.py`
(new `cond_channels_per_view` parameter on `ConditionalUNet`, default `4`,
threaded through to `AttrAwareSpatialPool`, since `cond_channels` alone
does not control the raw per-view channel count -- see the notebook's
intro cell for the full explanation and diff). This patch is a
prerequisite for Phase 2 as well (see below).

Checkpoint: `s1_tuk_native2ch_unet_best.pth`. Compare its
`s1_native2ch_validation_metrics.json` against notebook 04's baseline
`s1_validation_metrics.json`, focusing on `zncc`/`jsd`.

### 1.2. Real attributes instead of zero-filled (cheap, data already exists)

Replace the zero-filled `attrs` tensor with real per-view metadata already
sitting in `s1_patches_tuk_michel_provided`'s `attrs.json`
(acquisition_date, orbit_direction, relative_orbit_number). Tessa's
`AttrAwareSpatialPool` is designed to use real metadata to weight/fuse
views -- all-zeros defeats that mechanism entirely.

```python
import datetime as dt

LIDAR_SURVEY_DATE = dt.date(2024, 4, 16)  # Tuk LiDAR acquisition date

def build_real_attrs(s1_path, times, context_k):
    attrs_path = s1_path / 'attrs.json'
    attrs_list = json.load(open(attrs_path)) if attrs_path.exists() else []
    vecs = []
    for time_path in times:
        idx = int(time_path.stem[1:])  # 't0' -> 0
        a = attrs_list[idx] if idx < len(attrs_list) else {}
        if a.get('acquisition_date'):
            acq_date = dt.date.fromisoformat(a['acquisition_date'])
            age_days = (acq_date - LIDAR_SURVEY_DATE).days
            age_norm = age_days / 30.0
        else:
            age_norm = 0.0
        orbit_dir = 1.0 if a.get('orbit_direction') == 'ASCENDING' else 0.0
        rel_orbit = (a.get('relative_orbit_number') or 0) / 175.0
        vecs.append([age_norm, orbit_dir, rel_orbit, 0.0, 0.0, 0.0, 0.0, 0.0])
    return torch.tensor(vecs, dtype=torch.float32).flatten()
```
In `LidarS1Dataset.__getitem__`, replace:
```python
attrs = torch.zeros(8 * self.context_k, dtype=torch.float32)
```
with:
```python
attrs = build_real_attrs(s1_path, times, self.context_k)
```
Retrain with checkpoint name `s1_{REGION}_realattrs_unet_best.pth`, compare
ZNCC/JSD against the zero-attrs baseline.

### 1.3. Despeckle before feeding the model

Tests whether SAR's inherent multiplicative speckle noise (absent in
optical reflectance) is obscuring a genuine roughness-backscatter
relationship.

```python
from scipy.ndimage import median_filter

# inside the view loop, after reading sar but before dB conversion:
sar = np.nan_to_num(sar, nan=0.0, posinf=0.0, neginf=0.0)
sar = median_filter(sar, size=(1, 5, 5))  # despeckle each band spatially
sar = np.maximum(sar, 1e-12)
sar = 10.0 * np.log10(sar)
```
New checkpoint filename: `s1_{REGION}_despeckled_unet_best.pth`.

### 1.4. Backscatter vs. reflectance as a signal type -- not directly testable

Can't ablate this one (it's a sensor property, not a pipeline choice).
Existing evidence argues against it being the dominant factor: the raw,
pre-model correlation between Sentinel-1 backscatter and LiDAR roughness
(`compute_collocation_quality_stats` in notebook 06) was r=0.618-0.689 --
moderate-to-strong, computed independent of the deep model. That suggests
the physical signal itself is reasonably informative, and the bottleneck is
more likely in how the model processes it (supporting 1.1-1.3 over 1.4).

Optional follow-up: re-run `compute_collocation_quality_stats` on
`s1_patches_tuk_michel_provided` specifically (currently only run on the
own-patches dataset) to confirm the correlation holds there too.

### Phase 1 suggested order

Cheapest/most informative first: **(1.1) native channels -> (1.2) real
attrs -> (1.3) despeckling**. Each is a small code change plus one retrain;
comparing their individual effect on ZNCC (rather than combining all three
at once) tells you which factor actually matters most.

**Decision point after Phase 1:** if one or more of 1.1-1.3 closes most of
the ZNCC gap, the bottleneck was interface-level, not architectural --
stop here (or layer the winning changes together) rather than proceeding
to Phase 2. If ZNCC stays low across all three, that's evidence for
Phase 2: the architecture itself may not suit SAR's statistics.

---

## Phase 2: New model -- SAR-specific layer(s) on top of Tessa's architecture

Rather than another input-adapter ablation, this phase adds a genuinely
new component to the architecture, in the same spirit as Tessa's own
`AttrAwareSpatialPool` (a custom module addressing a domain-specific
property that a generic architecture wouldn't capture on its own).

### 2.1. Polarization-ratio feature layer (recommended first)

**Physical motivation:** the VH-VV difference in dB (equivalent to the
VH/VV ratio in linear power) is a well-established SAR roughness/
volume-scattering indicator, independent of either channel's absolute
calibration -- smoother surfaces are more co-pol dominated, rougher/
volume-scattering surfaces show relatively more cross-pol return. The
current model has to learn this relationship implicitly from raw VV/VH;
computing it explicitly gives the network a physically-motivated feature
it currently must discover on its own.

**Implementation sketch** (builds on the `cond_channels_per_view` patch
from 1.1, which makes the per-view channel count a real parameter instead
of a hardcoded `4`):

In `LidarS1Dataset.__getitem__`, after computing `sar` in dB (shape
`[2, H, W]`, order VV, VH):
```python
vv_db, vh_db = sar[0], sar[1]
pol_ratio_db = vh_db - vv_db  # VH - VV in dB == log10(VH/VV) scaled
sar_with_ratio = np.stack([vv_db, vh_db, pol_ratio_db], axis=0)  # [3, H, W]
```
Then interpolate/resize all 3 channels together (as the existing code
already does for the 2-channel case), and do **not** repeat to 4 channels.
Each view now contributes 3 channels instead of 2.

Model instantiation:
```python
model = ConditionalUNet(in_channels=1, cond_channels=4 * CONTEXT_K, attr_dim=8 * CONTEXT_K, base_channels=128, embed_dim=256, unet_depth=4, attention_variant=ATTENTION_VARIANT, cond_k=CONTEXT_K, cond_channels_per_view=3).to(DEVICE)
```
New checkpoint filename: `s1_{REGION}_polratio_unet_best.pth`.

Compare against whichever Phase 1 checkpoint performed best (not
necessarily the original zero-attrs/4-channel baseline), since Phase 2
should build on Phase 1's winning fix, not replace it.

### 2.2. Speckle-aware learned layer (more effort, if 2.1 isn't enough)

A learned, non-local-means-style block (aggregating statistics from
similar-looking neighboring pixels before the main encoder) as the
differentiable analog to classical SAR despeckling filters (Lee/Frost).
More novel than 1.3's fixed median filter, but more implementation risk.
Only pursue if 2.1 alone doesn't move ZNCC enough and there's time left.

### 2.3. Not currently feasible: InSAR coherence

SAR interferometric coherence (decorrelation between repeat passes) is
arguably the strongest SAR roughness indicator, but requires SLC
(phase-preserving) products and a coherence pipeline -- notebooks 01-03
currently acquire and process GRD amplitude data only. Flag as a
limitation/future-work item in the write-up rather than building it now,
unless there's time to scope out SLC access separately.

---

## Phase 3: Uncertainty / confidence map

**Motivation:** diffusion models are probabilistic samplers, not
deterministic regressors -- running the sampler multiple times on the same
input with different random noise seeds produces different outputs, and
that sample-to-sample variance is a legitimate, standard notion of
predictive uncertainty for diffusion models (generative/output
uncertainty, not full epistemic/weight uncertainty -- worth stating that
distinction explicitly in the write-up).

### 3.1. Build the ensemble

For a validation patch, run the sampler N times (suggest N=15-20) with
different random seeds, stack the results, and compute:
```python
predictions = torch.stack([sampler(model, scheduler, target.shape, condition, attrs, DEVICE) for _ in range(N)])
pred_mean = predictions.mean(dim=0)   # better point estimate than any single sample
pred_std = predictions.std(dim=0)     # per-pixel confidence/uncertainty map
```
Cost note: this multiplies inference cost by N per patch. Run the full
ensemble on a handful of example patches (reuse the existing
`example_patches` mechanism from notebook 04) for the visual confidence
map, not on the entire validation set.

### 3.2. Check calibration before trusting the map

The map is only meaningful if uncertainty correlates with actual error.
On a random subset of validation patches (suggest 30-50, not all 251, to
keep runtime reasonable):
```python
# per patch: pred_std (from 3.1) vs. actual |pred_mean - gt| per pixel
correlation = np.corrcoef(pred_std_flat, abs_error_flat)[0, 1]
```
A meaningful positive correlation (high uncertainty where the model is
actually wrong) is the deciding evidence for whether the confidence map is
a real result vs. just a visualization. **This correlation check, not the
map itself, is the answer to "does a confidence map work here."**

### 3.3. Visualization

Add an "Uncertainty (std)" row to the existing reconstruction-grid figure
(alongside S1 VV views / GT / Pred / Error / PDF), using `pred_std` per
example patch from 3.1.

### Which checkpoint to run this on

Can be applied to any trained checkpoint (Phase 1 baseline, Phase 1
winner, or Phase 2 model) -- doesn't depend on which phase finishes first.
Running it on whichever checkpoint currently performs best gives the most
meaningful calibration result.
