# Intervention A2: Tracking-failure retry / multi-hypothesis init

## Motivation

After five rejected interventions (C, B, B', A1, D1) the remaining
unaddressed failure is the **discrete single-frame catastrophic drift
event** at frame ~60-70, where per-frame ATE jumps ~14 cm in one frame
and never recovers. All five rejected interventions failed to reach
this event because they were either:

- **Mapping-side** (C, B, B') — runs AFTER the tracker commits a bad
  pose, cannot undo tracker drift.
- **Cross-keyframe** (A1) — under-constrained without ICP anchor.
- **Loss-formulation** (D1) — tightly coupled to the rest of the
  pipeline (weight schedule, densification).

A2 is a **tracker-side, single-frame, locally-scoped** intervention:
when the tracker's final loss indicates it probably landed in a local
minimum on the current frame, retry from multiple pose initializations
and keep the best-loss result.

Architecturally this is the **poor-man's DROID-VO substitute**. Instead
of a learned network providing a robust coarse pose estimate, search
over a small number of hand-chosen initializations. DG-SLAM's ATE is
~5 cm on BONN specifically because DROID-VO keeps the fine-refinement
photometric stage out of bad local minima. A2 attempts the same
outcome with a search rather than a learned net.

## Hypothesis (pre-registered)

**H0 (wiring)**: 25-frame smoke test on BONN person_tracking runs
without NaN/Inf. Retry fires on at least one frame of the smoke run
(else failure detection is mis-tuned). Final ATE within ±20% of
baseline.

**H1 (primary)**: BONN person_tracking, 100 frames stride=2, full
pipeline.
- Accept if ATE RMSE ≤ **14.00 cm** (≥30% improvement over 20.32 baseline).
- Marginal-accept if 14.00 < ATE ≤ 18.00 cm (confirm with H2).
- Reject if ATE > 20.32 + 2.00 = **22.32 cm** (regression budget).

**H2 (breadth)**: BONN balloon, 100 frames stride=2, full pipeline.
- Accept if ATE RMSE ≤ **40.00 cm** (≥30% improvement over 57.35 baseline).
- Marginal-accept if 40.00 < ATE ≤ 52.00 cm.
- Reject if ATE > 57.35 + 3.00 = **60.35 cm**.

**H3 (safety)**: Replica room0, 100 frames stride=1, full pipeline.
- Accept if ATE RMSE ≤ **1.50 cm** (baseline ~1.2 cm; static scene).
- Reject if ATE > 2.00 cm.

**Decision rule**: A2 is **accepted** only if H1 passes AND H2 does
not reject AND H3 does not reject. On H1 marginal, require H2 strict
pass. Any H3 regression auto-rejects.

## Design

### Failure detection signal

After the primary tracking pass returns `final_loss`, compare to the
moving median of recent frames. Trigger retry if:

```
final_loss > retry_loss_ratio_thresh * median(recent_losses[-10:])
```

Default `retry_loss_ratio_thresh = 2.5` — a frame where the tracker
settles at >2.5× the local median likely hit a bad local minimum.

Alternative / complementary signal: pose jump magnitude vs recent
motion pattern. Already covered by the existing motion-clamping
logic in pipeline.py (lines 476-500), so A2 uses loss-ratio as the
primary detector and lets motion-clamping remain as a downstream
safety net.

The first ~5 frames have no meaningful recent-loss history; skip
retry in that warmup phase (retry always has finite overhead, no
benefit when baseline is already the only init).

### Retry hypotheses

Four init poses tried (including primary):

| # | Init                          | Rationale |
|---|-------------------------------|-----------|
| 1 | velocity-init (baseline)      | What tracker already does |
| 2 | previous pose (zero velocity) | Stationary hypothesis — helps when motion estimator was wrong |
| 3 | damped velocity (0.5× delta)  | Scene where motion was decelerating |
| 4 | velocity + small SE(3) pert.  | Escape symmetric local minima |

Hypothesis 4 uses a deterministic perturbation (fixed seed per frame)
of ±3° yaw and ±2cm translation along the optical axis. Deterministic
so runs are reproducible for pre-registered protocol.

For each hypothesis, call `tracker.track(...)` and record `final_loss`.
Keep the pose from the hypothesis with lowest final loss. If all
hypotheses exceed the baseline by a margin (suggesting the tracker
genuinely cannot converge on this frame), fall back to primary — the
alternatives aren't helping, no need to make it worse.

### When retry does NOT fire

- Warmup: first 5 frames (insufficient history).
- Loss ratio below threshold: primary is probably fine.
- `use_tracking_retry: false` in config (disables feature entirely).

### Cost

Retry fires only on flagged frames. If ~10% of frames are flagged
(reasonable), adds ~30% to total tracking wall-clock. Acceptable for
this experiment.

## Implementation plan

### Files

1. `dynlang_slam/slam/pipeline.py` — retry loop around `self.tracker.track()`
   - New config reads: `cfg.slam.tracking.use_tracking_retry`,
     `retry_loss_ratio_thresh`, `retry_num_hypotheses`, `retry_warmup_frames`
   - New method `_retry_track(...)` that generates hypotheses, calls
     tracker for each, returns best

2. `configs/default.yaml` — add retry block under `slam.tracking`

3. `scripts/smoke_tracking_retry.py` — smoke test verifying
   - retry fires on at least one hard frame
   - ATE finite
   - no NaN/Inf

### Decoupling from mapper

A2 only touches pipeline tracking logic. No mapper changes, no
densification changes, no loss-formulation changes. This is
specifically designed to sidestep the coupling failures that killed
C, B, B', A1, D1.

## Why this is not the same failure pattern as prior five

| | C | B | B' | A1 | D1 | **A2** |
|---|---|---|---|---|---|---|
| Mapping-side?          | partial | yes | yes | yes (BA) | no | **no** |
| Cross-keyframe?        | no  | no  | no  | yes | no | **no** |
| Changes loss formula?  | yes | no  | no  | no  | yes | **no** |
| Adds DOF?              | no  | no  | no  | yes | no | **no** |
| Upstream assumption?   | SplaTAM 200k | DG DROID-VO | dense GT | MonoGS ICP | DG DROID-VO | **none** |
| Fires on drift event?  | no  | no  | no  | partial | no | **yes** |

A2 is locally scoped to the single frame currently being tracked.
It cannot affect mapping, densification, cross-keyframe consistency,
or map growth. Failure mode, if any, would have to come from retry
picking the WRONG hypothesis — but we detect that via final_loss,
which is the same signal the primary tracker uses, so the detection
is self-consistent.

## Permitted tuning before outright reject (per protocol)

If H1 marginal-fails or fails by <5 cm:
- Try `retry_loss_ratio_thresh ∈ {2.0, 3.0}` once each (tighter / looser)
- Try `retry_num_hypotheses ∈ {3, 6}` once each

Total retry-tuning budget: 4 additional benchmark runs (= ~25 min GPU).
If none lands H1, reject outright.

## Budget

- Protocol + implementation: 30 min
- Smoke test: 2 min
- H1/H2/H3: ~20 min combined
- Tuning (if needed): 25 min
- Total: ~80 min

## Result tracking

- Smoke test log → `research/experiments/tracking-retry/smoke_log.txt`
- H1 → `research/experiments/tracking-retry/h1_person_tracking.txt`
- H2 → `research/experiments/tracking-retry/h2_balloon.txt`
- H3 → `research/experiments/tracking-retry/h3_replica_room0.txt`
- Tuning runs (if any) → `research/experiments/tracking-retry/h1_tune_*.txt`
- Final verdict → `research/experiments/tracking-retry/analysis.md`
