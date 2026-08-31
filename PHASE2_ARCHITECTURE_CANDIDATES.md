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

**Superseded by the `pcrtc/06` result below -- see the 2026-08-31 update
section for the current picture.** Original framing, kept for the
historical record of why these candidates were ranked this way:

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

## Update, 2026-08-31: `pcrtc/06` changes the picture substantially

`notebooks/pcrtc/06_train_realattrs.ipynb` (PC-RTC + real per-view attrs,
repeated 4-channel VV/VH, no native2ch) is now the best result in the
entire study -- see `CONCEPTS.md`'s decomposition section for the full
numbers. This directly changes both "still broken" problems above:

1. **Problem #2 (variance compression) is essentially resolved by data
   source + real attrs alone**, with no new architecture needed:
   `pred_std/gt_std` = 1.02 (vs. 0.66 for the plain PC-RTC baseline), and
   the ensemble-uncertainty calibration check jumps to R²=0.880. This
   removes the strongest justification for candidate #2
   (distribution-matching loss) -- there may be little left for it to fix.
2. **Problem #1 (ZNCC gap) shrinks substantially but doesn't close**:
   ZNCC=0.519 vs. Tessa's ~0.74-0.78, now roughly a 1.4-1.5x gap (down
   from ~2.6-2.7x). Candidate #1 (DEM conditioning) is still the
   best-targeted option for this remaining gap, but the size of the
   problem it needs to solve is much smaller than when this ranking was
   first written.

**Revised recommendation**: if a new architecture experiment is still
worth doing given the time left, build it on top of **`06`'s config**
(PC-RTC + real attrs) as the base, not `03`'s plain baseline -- `06` is
the strongest known starting point on any config tried, independent of
this decision. **DEM conditioning (#1) remains the top candidate**, since
the remaining problem it targets (ZNCC) is the one still open; candidate
#2 (distribution-matching loss) is now lower priority than originally
ranked, since the problem it was designed to fix looks largely solved
already by `06`'s data-source/attrs combination.

This also means the case for *not* pursuing a new architecture at all is
stronger than before -- `06` alone may already be a strong enough
headline result for the dissertation (best in the entire study, closes
most of the S2 gap, near-perfect uncertainty calibration) that the
marginal value of a new architecture layer is smaller than it looked when
the gap was 2.6-2.7x. Worth putting this framing to Michel directly rather
than assuming more experimentation is automatically better.

## Time-boxed recommendation, added 2026-08-31

Deadline runway changed the calculus, so recording both answers rather
than one fixed ranking. Note: written before the `06` update above --
the *choice* of candidate (DEM conditioning vs. distribution-matching
loss) should now also weigh the `06` update's point that variance
compression looks largely solved already, independent of days remaining.

**At ~15 days remaining** (deadline read as 2026-09-15 from 2026-08-31):
recommended **#2, distribution-matching loss** first. Reasoning: it's a
pure training-loop change (new loss term alongside the existing masked
MSE) with no new data-acquisition dependency, so it's the fastest path to
a result under a tight timeline. It targets problem #2 (variance
compression) directly. #1 (DEM conditioning) was judged too expensive to
start safely at 15 days once its full cost is counted (coverage check +
acquisition/collocation + new encoder branch + training + debugging
buffer). *Revised in light of `06`: with variance compression now largely
resolved by data source + real attrs alone, #2's justification is weaker
than when this was written -- at 15 days, reporting `06` as the
dissertation's headline result may be the better use of the remaining
time than either architecture candidate.*

**At ~22 days remaining** (user-stated timeline from 2026-08-31, i.e.
target date around 2026-09-22 -- see note below on the deadline
discrepancy): recommendation is **#1, DEM conditioning**, built on top of
`06`'s config. Reasoning: the extra ~week is enough runway to absorb its
higher setup cost, and it targets the remaining, now better-quantified
gap (ZNCC 0.519 vs. Tessa's 0.74-0.78, ~1.4-1.5x). It's also the option
that's a genuine new architecture layer, matching the original ambition
("i really want to get something new, just as one layer tessa used"),
rather than a loss-function tweak. Rough day budget proposed:

- Days 1-2: DEM coverage check for Tuktoyaktuk (Copernicus GLO-30 or
  ArcticDEM) + acquisition/collocation.
- Days 3-5: implement the conditioning branch (small encoder, fused
  similarly to how `AttrAwareSpatialPool` already fuses S1 views) --
  code drafted, see chat log / notebook `dem/01-02`.
- Days 6-9: training + debugging buffer (architecture changes on this
  project have consistently hit at least one shape/loading bug). At
  ~170min/full run (measured on `06`: 100min train + 70min eval), even
  4-5 failed attempts costs under a day of GPU time -- compute isn't the
  bottleneck, implementation/debugging is.
- Days 10-11: evaluation, confidence map, comparison against `06`.
- Remaining ~10 days: writing, with buffer to fall back to reporting `06`
  as the dissertation's core finding if the DEM path underperforms or
  runs long.

**Deadline discrepancy, unresolved:** this project's prior confirmed
deadline (2026-08-30, see `CONCEPTS.md`/memory) was **2026-09-15**, i.e.
15 days from 2026-08-31, not 22. The user stated "22 days" on 2026-08-31
without correcting this when the discrepancy was flagged back to them.
Treating 22 days as the current operative number for this recommendation,
but the actual date should be confirmed/reconciled before finalizing which
recommendation (15-day vs. 22-day version above) actually applies.
