# Intervention A2 (tracking-failure retry / multi-hypothesis init): REJECTED

## Verdict

**REJECT.** Not because the retry picks bad hypotheses when fired, but
because the **loss-ratio failure detector does not fire on the drift
event it was designed to catch.** The core premise — that tracker
final_loss is a reliable proxy for pose correctness — is empirically
false in our photometric 3DGS tracking regime.

## Results

| Benchmark                          | ATE (cm) | Budget               | Verdict     |
|------------------------------------|----------|----------------------|-------------|
| Smoke (25f, stride=2)              | 13.88    | <50 cm, no NaN       | PASS (wiring) |
| H1 person_tracking (100f)          | 21.59    | accept≤14, reject>22.32 | not accept, not reject (marginal) |
| H2 balloon (100f)                  | 55.69    | accept≤40, reject>60.35 | not accept, does not reject |
| H3 Replica room0 (100f, A2 on)     | 13.66    | accept≤1.50, reject>2.00 | numerically fails budget, but A2 not at fault |
| H3 Replica room0 (100f, A2 OFF)    | 13.73    | (baseline)           | confirms A2 neutral on Replica |
| Tune thresh=2.0 (permitted)        | 31.34    | — | 0 retries, worse than baseline (run-to-run noise) |
| Tune thresh=3.0 (permitted)        | 26.81    | — | 0 retries, worse (noise) |
| Diag thresh=1.3 (out-of-protocol)  | 37.06    | — | 3 retries fire, ATE WORSE |

## Why REJECT

### 1. The detector does not fire on the target event

A2 was designed to fire on the known discrete drift event at BONN
person_tracking frame ~60–70. Measured per-frame tracker losses over
the full 100-frame run:

```
loss median = 0.0599 – 0.0846   (depending on run)
loss max    = 0.1015 – 0.1155
max/median  = 1.37× – 1.69×     (WELL below the 2.5× trigger)
```

At frame 70 — the center of the drift event where per-frame ATE jumps
from ~15 cm to ~26 cm — the tracker's final_loss is 0.0737. The
running median of the preceding 10 frames is ~0.05–0.07, for a ratio
of roughly 1.0–1.5×. The retry trigger requires 2.5×. **It cannot
fire there.**

Even under the permitted tuning budget, lowering the threshold to
`2.0` (permitted) still yields **0 retries over 99 frames.** Lowering
it further to `3.0` (permitted, the other permitted value) also yields
**0 retries.**

An out-of-protocol diagnostic at `thresh=1.3` produced 3 retry events
(all at frames 91, 92, 94) — none on the actual drift event. **The
drift event has no loss signature at all.**

### 2. Even when retry fires, it doesn't help

The out-of-protocol `thresh=1.3` run fired 3 retries and two of them
selected alternate hypotheses (`perturb`, `half_vel`). Final ATE was
**37.06 cm — worse** than both the A2-disabled baseline (~20 cm) and
the A2-enabled baseline (~21.59 cm).

Mechanism: the alternates find hypotheses with slightly lower final
loss, but lower loss ≠ better pose. The tracker's photometric loss
measures *local* consistency between rendered Gaussians and the image.
A pose that slightly-wrongly places the camera in a region of dense
Gaussians can look "good" by this metric while being further from GT
than the primary. The retry picks the *least-wrong-looking* pose
rather than the *most-accurate* pose.

### 3. H3 safety gate: A2 neutral, regression pre-exists

The pre-registered H3 baseline was "~1.2 cm". At the current full
pipeline config (680×1200 Replica resolution, dynamic+language enabled),
measured H3 ATE is 13.66 cm with A2 on and **13.73 cm with A2 off** —
a difference of 0.07 cm, well within noise. `retry_fired=False` on
all 99 Replica frames, so A2 is cleared of contributing to this
regression. The protocol's H3 baseline was stale; it was calibrated on
an older/smaller/static-only configuration and should be re-established
independently before being used as a gate.

Taking A2 as the proximate H3 cause (as the strict protocol reading
would) is unsupported by the data.

## Root cause: loss magnitude is a weak proxy for pose correctness

This is a genuinely new diagnostic finding worth preserving:

> In photometric 3DGS tracking, the tracker's final photometric loss
> is **not a reliable failure signal.** When the tracker lands in a
> locally-consistent-but-globally-wrong pose minimum, the rendered
> Gaussians still align reasonably with image pixels (because the map
> itself was built from those Gaussians, and they re-project similarly
> under a range of nearby poses). The photometric loss is small.
> ATE is large.

DROID-SLAM and DROID-VO work around this by maintaining dense optical
flow correspondences that can disagree with a photometrically-consistent-
but-geometrically-wrong pose. Our tracker has no such external check,
so loss-magnitude-based failure detection is structurally limited.

Any future retry / failure-recovery intervention must use a signal
that **disagrees with the tracker's own objective**. Candidates:
- Depth residual from independent geometric warp
- Feature-correspondence disagreement (ORB / learned descriptor)
- Pose velocity outlier (magnitude / direction) vs local KF graph
- Rendered alpha coverage collapse

## Why A2 is in the same meta-pattern as C/B/B'/A1/D1

| | Recipe assumption | What our baseline lacks |
|---|---|---|
| C (silhouette gate) | SplaTAM 200k-init, 40-iter mapping | sparse 16k init |
| B (depth warp) | DROID-VO upstream pose | velocity-init only |
| B' (PIPs mask) | dense-GT-pose-like quality | tracked poses drift |
| A1 (local BA) | MonoGS ICP anchor + large KF graph | no ICP, small window |
| D1 (loss weights + RGB mask) | DG-SLAM DROID-VO + retuned schedule | time-varying schedule load-bearing, densify coupled |
| **A2 (retry)** | **reliable loss-based failure signal** | **tracker loss is not that signal** |

All six rejections share the same meta-pattern: **the recipe imported
from literature works only in combination with an upstream capability
our baseline lacks.** A2's missing upstream capability is the most
fundamental: we are trying to build a failure recovery on top of a
signal that doesn't exist for our system.

## Preserved side-findings

- Loss magnitude does not correlate with pose error at drift events.
  The drift at frame 60–70 shows `loss/median ≈ 1.0`, not a spike.
- BONN 100-frame run-to-run ATE variance is large (~±10 cm in our
  measurements: 16.30 / 21.59 / 26.81 / 31.34 / 37.06 across different
  seeds / config variants, with identical A2-off regions). Any future
  ATE comparison must budget for this.
- When retry fires and picks an alternate hypothesis, ATE gets worse.
  This suggests the search space (prev pose, damped velocity, SE(3)
  perturbation) is not capturing the correct answer either — another
  sign that the real failure is upstream (map quality / depth / tracker
  convergence) rather than pose-initialization.
- Replica-Room0 at full-res 680×1200 + dynamic + language currently
  runs at ~13.7 cm. The pipeline's "~1.2 cm on Replica" baseline is
  stale / from a different config. A fresh Replica baseline should be
  re-established before using it as a safety gate.

## Decision: fully revert A2, keep config knobs, document rejection

- `configs/default.yaml`: set `use_tracking_retry: false`.
- Keep all retry_* config entries (documented REJECTED, protocol
  preserved for future interventions that might re-use the knobs).
- `research/research-state.yaml`: append A2 rejection record.
- Code stays in `dynlang_slam/slam/pipeline.py` — the retry machinery
  is dormant when `use_tracking_retry=false`. It is useful to keep
  because the failure-detector abstraction is general; a future
  intervention could replace the loss-ratio trigger with a stronger
  signal (depth warp residual, feature match disagreement) without
  re-implementing the hypothesis-generation machinery.

## Next intervention candidate

Per `research-state.yaml`, remaining candidates are:
- **B3** (Tukey M-estimator in tracker). Simpler than A2; changes loss
  robustification rather than failure detection. Does not require a
  reliable failure signal.
- A new candidate emerged from this rejection: **external failure
  detector** — compute depth warp residual independently of the
  tracker's objective, use as a retry trigger. Higher cost but
  addresses the root cause identified here.
