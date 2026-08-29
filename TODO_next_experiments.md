# TODO: Diagnosing the Sentinel-1 vs Sentinel-2 performance gap

Context: `s1_patches_tuk_michel_provided` evaluation gave ZNCC=0.166 vs
Tessa's Sentinel-2 reference of ~0.74-0.78 -- comparable RMSE/bias, but much
weaker spatial correlation. Four candidate explanations, with a concrete
experiment for each. See `CONCEPTS.md` for full background and the
S1-vs-S2 comparison table.

## 0. Coarse-resolution evaluation -- DONE (2026-08-29)

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
resolution-limit artifact.** This rules toward experiments 1-3 below
(model-side bottlenecks: channel repetition, zero-filled attrs, speckle)
rather than "S1 is just coarser so of course it looks worse." Saved to
`/cs/student/project_msc/2025/aibh/jiayiche/s1_training_outputs/s1_coarse_vs_fine_comparison.json`.

Next: run experiments 1-3 in the suggested order below.

## 1. Real attributes instead of zero-filled (cheap, data already exists)

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

## 2. Native 2-channel architecture (no channel repetition)

Tests whether feeding `[VV, VH, VV, VH]` (out-of-distribution for filters
trained on real 4-channel R/G/B/NIR) is itself degrading performance,
independent of whether the underlying signal is informative.

**Change 1** -- in `LidarS1Dataset.__getitem__`, delete:
```python
sar_tensor = sar_tensor.repeat(2, 1, 1)
```
**Change 2** -- update model instantiation:
```python
model = ConditionalUNet(in_channels=1, cond_channels=2 * CONTEXT_K, attr_dim=8 * CONTEXT_K, base_channels=128, embed_dim=256, unet_depth=4, attention_variant=ATTENTION_VARIANT, cond_k=CONTEXT_K).to(DEVICE)
```
**Change 3** -- new checkpoint filename: `s1_{REGION}_native2ch_unet_best.pth`.

## 3. Despeckle before feeding the model

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

## 4. Backscatter vs. reflectance as a signal type -- not directly testable

Can't ablate this one (it's a sensor property, not a pipeline choice).
Existing evidence argues against it being the dominant factor: the raw,
pre-model correlation between Sentinel-1 backscatter and LiDAR roughness
(`compute_collocation_quality_stats` in notebook 06) was r=0.618-0.689 --
moderate-to-strong, computed independent of the deep model. That suggests
the physical signal itself is reasonably informative, and the bottleneck is
more likely in how the model processes it (supporting 1-3 over 4).

Optional follow-up: re-run `compute_collocation_quality_stats` on
`s1_patches_tuk_michel_provided` specifically (currently only run on the
own-patches dataset) to confirm the correlation holds there too.

## Suggested order

Cheapest/most informative first: **(2) native channels -> (1) real attrs
-> (3) despeckling**. Each is a small code change plus one retrain;
comparing their individual effect on ZNCC (rather than combining all three
at once) tells you which factor actually matters most.
