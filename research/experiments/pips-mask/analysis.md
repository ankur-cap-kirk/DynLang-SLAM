# Analysis: CoTracker-based dynamic mask (intervention B')

## Result summary

| Sequence        | Mode    | Baseline ATE | PIPs ATE | Δ       | Accept threshold | Verdict |
|-----------------|---------|-------------:|---------:|--------:|-----------------:|:--------|
| balloon         | full    | **57.35**    | **57.69**| +0.34   | ≤ 40.00 (H1)     | **FAIL**|
| balloon         | static  | ~58          | 53.68    | -4.32   | —                | —       |
| balloon         | dynamic | ~55          | 55.93    | +0.93   | —                | —       |
| person_tracking | full    | **20.32**    | **19.83**| **-0.49** | ≤ 22.32 (H2)  | **PASS**|
| person_tracking | static  | 28.36        | 32.15    | +3.79   | —                | —       |
| person_tracking | dynamic | 21.85        | 24.38    | +2.53   | —                | —       |

- **H0 (diagnostic)** — passed earlier: `smoke_pips_mask.py` showed PIPs
  adds +68.73 pp of dynamic area over YOLO on balloon with GT poses.
  Threshold sweep located the noise/signal knee at 25 px.
- **H1 (primary)** — **FAIL**. Balloon full-mode ATE is essentially
  unchanged (57.69 vs 57.35 cm). The class-agnostic signal that showed up
  clearly with GT poses did not translate into MAP improvement at the
  tracked-pose regime the actual SLAM loop runs in.
- **H2 (safety)** — **PASS**. person_tracking full-mode ATE dropped by
  0.49 cm (19.83 vs 20.32), comfortably inside the +2 cm regression budget.

## Diagnosis

The gap between H0 (strong PIPs signal with GT poses) and H1 (no ATE
improvement with tracked poses) is the interesting finding.

1. **The mask IS firing.** Balloon avg dynamic rises from 11.3% (YOLO
   only, dynamic mode) to 12.0% (YOLO + PIPs, full mode), and on
   person_tracking it rises from 10.9% → 25.7%. PIPs is contributing
   additional dynamic pixels.
2. **It isn't helping the balloon map.** A 0.7 pp bump in dynamic coverage
   on balloon is too small to explain the full 17 cm gap to the 40 cm
   budget. The balloon occupies on the order of 5-10% of the frame at
   peak, and our threshold sweep found its signal at thresh_px = 25. With
   tracked (noisy) poses, many balloon-pixel residuals likely fall below
   25 px on most frames, because the bias cancellation via
   median-correction is less clean when the background residuals are
   themselves elevated by pose noise.
3. **Person_tracking's 25.7% dynamic is suspicious.** That's more than
   double the YOLO-only rate. PIPs is flagging regions beyond the person,
   yet ATE still improves. This suggests the extra flagged regions are
   either (a) actually dynamic (e.g., motion-blur or parallax-inconsistent
   background), or (b) harmless false positives in already-well-mapped
   areas. The map wasn't hurt by losing them.
4. **The tracker never saw the PIPs mask.** Per the design (mask computed
   AFTER tracking), the balloon still corrupts the current frame's pose
   optimization. This is arguably the bigger leak on balloon — the tracker
   drifts ON the balloon, then no amount of clean mapping can recover.

## Verdict under pre-registered protocol

Protocol clause: *"If H1 fails but H0 and H2 pass → tune threshold / grid
density before giving up."*

H0 passed (offline diagnostic), H2 passed (safety), H1 failed (primary).
Per protocol this is a **tune-before-reject** branch, not an outright
rejection.

Two tuning directions are well-motivated by the diagnosis:

- **Lower `thresh_px`** (e.g., 15 px instead of 25): the 25 px knee was
  chosen from GT-pose residuals, but tracked poses add noise. A lower
  threshold may recover balloon pixels at the cost of more false
  positives — but the person_tracking H2 budget has 2.5 cm of headroom,
  so some regression is tolerable if balloon improves.
- **Denser `grid`** (e.g., 30×30 = 900 points instead of 20×20 = 400):
  more seed points means better spatial coverage of small dynamic
  objects, and the max-over-window aggregation means a single hit can
  flag an entire dilation neighborhood.

Either single knob-turn is ~15 min per full run × 2 sequences. Budget is
cheap relative to intervention cost.

## Open question (not addressed by tuning)

The deeper question is whether a mask that only affects mapping —
without also guarding the tracker — can ever close balloon. If the
tracker drifts on balloon pixels during the current frame, the pose it
publishes is already wrong, and every downstream Gaussian we add is
seeded with that wrong pose. Even a perfect map-side mask can't unbreak
that. This is an argument for eventually letting the PIPs mask also feed
the tracker — but only after the mask quality question is settled
standalone.

## Tuning attempt (thresh_px = 15)

Per pre-registered protocol, ran the lower-threshold branch first.

| Sequence        | Mode    | thresh=25 | thresh=15 | Δ      |
|-----------------|---------|----------:|----------:|-------:|
| person_tracking | dynamic | 24.38     | 33.07     | +8.69  |
| person_tracking | **full**| **19.83** | **27.10** | **+7.27** |

Full ATE 27.10 cm at thresh=15 vs H2 budget 22.32 cm → **H2 now FAILS**.

Per-frame diagnosis (dynamic mode at thresh=15):
- Frame 20: dyn=76.8%, Frame 40: dyn=67.5%, Frame 60: dyn=83.6%, Frame 80: dyn=61.6%
- Avg dyn% on person_tracking: 10.9% (thresh=25) → 49.2% (thresh=15)

The lower threshold doesn't just catch more dynamic pixels — it floods
the mask with false positives from pose-error noise on static
background. At 70-80% frame coverage the mask is erasing the map.

Balloon run at thresh=15 was skipped: once H2 safety breaks, no
single config satisfies both H1 and H2, so H1 outcome is moot.

## Why grid-density tuning also won't save this

Protocol listed two tuning directions: lower threshold and denser grid.
Threshold tuning backfired because the failure mode is
*over-classification* (noise crossing a too-low threshold), not
*missed detection*. A denser 30×30 grid supplies MORE query points
whose noise can cross the threshold, which compounds the problem — it
does not address it. Skip.

## Final verdict: REJECT intervention B'

Same discipline as B and C:

- H0 passed (offline diagnostic with GT poses): PIPs signal exists.
- H1 failed at thresh=25 (balloon 57.69 vs 40 cm budget) and not
  recoverable at thresh=15 (would require H2-breaking settings).
- H2 marginal at thresh=25 (-0.49 cm, inside budget) and broken at
  thresh=15 (+7.27 cm, outside budget).
- No single configuration jointly satisfies H1 and H2.

The intervention produced a real signal in the diagnostic (H0) that
did not translate to ATE improvement in the actual SLAM loop. Two
hypotheses for why:

1. **Pose-error floor.** Tracked-pose residuals carry more noise than
   GT-pose residuals; median-correction cancels the DC component but
   not the per-point noise, so a robust-to-both-regimes threshold may
   not exist.
2. **Tracker coupling.** PIPs runs AFTER tracking by design. On
   balloon, the tracker drifts ON balloon pixels in the current frame
   before the mask can act. A map-only mask cannot undo that drift.

Hypothesis (2) is the deeper issue and is the structural argument for
the next intervention (**A1: Local BA over the keyframe window**),
which retroactively smooths bad poses using evidence from later
keyframes. A1 addresses the drift-event failure mode directly, which
this intervention's mapping-side fix could not reach.

## Disposition

- Code retained on `feat/pips-dynamic-mask` branch for reference.
- `dynamic.pips.enabled` flipped to `false` in default config.
- `thresh_px` reverted to 25 (less-broken variant if ever re-enabled).
- Branch merged to main with `--no-ff` to preserve the rejection as a
  first-class commit (same discipline as B and C).
- `research-state.yaml` updated: B' rejected, next = A1 Local BA.
