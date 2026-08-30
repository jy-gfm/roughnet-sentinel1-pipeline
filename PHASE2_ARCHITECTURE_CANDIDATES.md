# Phase 2 Architecture Candidates: Ranked

Companion to `TODO_next_experiments.md`'s Phase 2 section. That file has
the original two candidates with implementation sketches; this file adds
two more (surfaced from reading `A Diffusion-Based Framework for
Terrain-Aware Remote Sensing Image Reconstruction`, arXiv:2504.12112) and
ranks all four against what's actually still broken after Phase 1 + Phase 4.

**Status: not yet started.** Per `TODO_next_experiments.md`'s Phase 2
status note, this decision is deliberately paused pending Michel's input
-- this document exists so that conversation can happen with a concrete,
ranked menu in hand, not so any of these get built unilaterally.

## What's actually still broken (the target these candidates address)

After Phase 1 (raw-SAFE preprocessing ablations) and Phase 4 (Planetary
Computer RTC data source), two distinct problems remain unsolved:

1. **Spatial pattern fidelity (ZNCC) is still well below Tessa's Sentinel-2
   reference**, even after the best fix found so far (PC-RTC baseline,
   ZNCC=0.286 vs. Tessa's ~0.74-0.78 -- still a ~2.6-2.7x gap).
2. **Predicted variance is systematically compressed relative to ground
   truth** (`pred_std` consistently lower than `gt_std`) across *every*
   experiment run so far, on both data sources -- this hasn't improved
   with any preprocessing or data-source fix tried.

Each candidate below is evaluated against which of these two problems it
actually targets, since a new architecture layer is only worth the
implementation cost if it has a plausible mechanism for fixing one of
them specifically.

## Ranked candidates

### 1. DEM conditioning, ControlNet-style (new, highest priority)

**What**: feed an independent, coarse Digital Elevation Model (e.g.
Copernicus GLO-30 or ArcticDEM, ~10-30m resolution -- much coarser than
the 1m LiDAR target, so this isn't circular) as an additional conditioning
input alongside the Sentinel-1 imagery, similar to how SatelliteMaker
(arXiv:2504.12112) uses DEM as a ControlNet-style conditioning signal to
"enhance spatial consistency and geographic accuracy... especially in
areas with complex terrain."

**Why it's ranked first**: it directly targets problem #1 (ZNCC), which
has been the single most persistent bottleneck across every experiment in
this project. The model currently has to infer large-scale terrain
structure purely from noisy SAR backscatter; giving it an independent,
lower-noise terrain-shape signal could plausibly help it get large-scale
spatial pattern right even where the SAR signal alone is ambiguous.

**Feasibility**: moderate. Needs a new conditioning branch (e.g. a small
encoder for the DEM channel, concatenated or added similarly to how
`AttrAwareSpatialPool` already fuses S1 views) and a new data-acquisition
step (downloading/collocating a coarse DEM product for the study area --
lighter than the Sentinel-1 acquisition pipeline, since DEM products are
static, no temporal matching needed). Caveat: Arctic coastal DEM products
can have gaps/errors in some areas -- worth a quick coverage check before
committing, same as was done for Planetary Computer RTC data.

### 2. Distribution-matching loss during training (new, second priority)

**What**: add an explicit loss term during training that penalizes
distributional mismatch between predicted and ground-truth outputs (the
underlying idea behind SatelliteMaker's VGG-Adapter/distribution loss,
adapted from RGB feature-space matching to single-channel elevation
statistics -- e.g. matching within-patch std/histogram directly rather
than a full VGG perceptual loss, since there's no "style/color" dimension
in a single-channel elevation map).

**Why it's ranked second**: directly targets problem #2 (variance
compression), which has been *measured* (JSD is already an eval metric)
but never *optimized for* during training -- the model has never been
explicitly penalized for under-spreading its predictions. This is a
plausible, targeted fix for a specific, well-documented, recurring
failure mode.

**Feasibility**: moderate. Pure training-loop change (new loss term,
combined with the existing masked MSE) -- no new data acquisition needed,
which makes it cheaper than #1 in that respect. Main risk is tuning the
loss weighting so it doesn't destabilize the existing reconstruction loss.

### 3. Learned polarization-gating module (from earlier discussion, third priority)

**What**: a small new `nn.Module` that computes the physical VH-VV
polarization ratio and passes it through a learned conv/MLP to produce a
per-pixel gating map, modulating the VV/VH features before the rest of
the network -- genuinely new architecture (not just an extra input
channel), physically motivated by the same VH/VV roughness indicator
already documented in `TODO_next_experiments.md` 2.1.

**Why it's ranked third**: doesn't have as direct a mechanistic link to
either of the two specific remaining problems as #1 or #2 -- it's a
reasonable, physically-motivated architecture addition, but more
exploratory than "targeted fix for a known, measured failure mode."

**Feasibility**: moderate, similar effort to #1/#2. No new data needed
(VV/VH already available).

### 4. Speckle-aware non-local learned block (existing candidate, lowest priority)

**What**: the most ambitious option already sketched in
`TODO_next_experiments.md` 2.2 -- a non-local-means-style learned block,
the differentiable analog of classical SAR despeckling filters (Lee/
Frost/Refined Lee).

**Why it's ranked last**: the despeckling ablation (notebook 10) already
returned a clear *negative* result with a simple filter -- ZNCC and RMSE
both got worse, suggesting the noise/texture separation problem is
genuinely hard, not just under-engineered. A learned version could in
principle do better (it's optimized end-to-end for the actual task rather
than a generic denoising objective), but there's no strong evidence yet
that this specific direction is promising, and it's also the highest
implementation/debugging risk of the four options.

## Not included: LoRA

Considered after reading SatelliteMaker (arXiv:2504.12112) in full --
ruled out, not just deprioritized. LoRA's entire value proposition is
cheaply adapting a *large pretrained foundation model* (SatelliteMaker
builds on Stable Diffusion) by freezing most weights and injecting small
trainable low-rank matrices. Tessa's `ConditionalUNet` is a small
architecture trained from scratch on this project's own dataset -- there
is no large pretrained checkpoint underneath it to adapt, so LoRA has no
mechanism to help here. See `CONCEPTS.md` for the full reasoning.

## Recommended next step

Bring this ranked list to Michel alongside the Phase 4 results (per the
plan already in `TODO_next_experiments.md`) rather than starting
implementation unilaterally -- particularly because #1 (DEM conditioning)
introduces a new external data dependency worth confirming is a sensible
direction before investing acquisition + implementation time in it.
