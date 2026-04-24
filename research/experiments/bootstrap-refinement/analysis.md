# BR1 Analysis: Bootstrap-Window Joint Pose Refinement — REJECTED

**Date**: 2026-04-24
**Protocol**: `research/experiments/bootstrap-refinement/protocol.md` (pre-registered 2026-04-23)
**Evaluator**: `scripts/br1_eval.py`
**Commit state at run**: BR1 implemented via `SLAMPipeline.maybe_run_bootstrap` +
`_run_bootstrap_refinement`, config block `slam.bootstrap` added to
`configs/default.yaml` (default off), triggered in eval by `pipeline.bootstrap_enabled = True`.

## Pre-registered accept criteria

| Sequence           | Baseline (aligned) | Accept if aligned ATE ≤ |
|--------------------|-------------------:|------------------------:|
| H1 person_tracking |             7.56   |                **5.50** |
| H2 balloon         |             8.55   |                **6.00** |
| H3 replica room0   |             3.68   |                **4.10** |

**ACCEPT** requires all three. **REJECT** on ANY of: H3 > 4.10, H1 improves
by < 0.5 cm, both H1 AND H2 regress.

## Results

| scene                      | aligned ATE (cm) | baseline | Δ        | budget | verdict |
|----------------------------|-----------------:|---------:|---------:|-------:|:--------|
| H1 person_tracking         |             9.37 |     7.56 | **+1.81**|   5.50 | **FAIL** |
| H2 balloon                 |            13.89 |     8.55 | **+5.34**|   6.00 | **FAIL** |
| H3 replica room0           |             6.31 |     3.68 | **+2.63**|   4.10 | **FAIL** |

**Verdict: REJECT.** Three REJECT conditions triggered simultaneously:
- H3 safety regression (+2.63 cm on static scene, ~71% worse than baseline).
- H1 regresses (not improves by ≥ 0.5 cm) — opposite of hypothesis direction.
- Both H1 and H2 regress.

## BR1 event diagnostics

The BR1 event itself barely moved poses:

| scene | BR1 fired? | K_refined | max Δt | mean Δt | max Δrot |
|-------|:----------:|----------:|-------:|--------:|---------:|
| H1    | **NO** (too few KFs at frame 15) | — | — | — | — |
| H2    | yes        | 2         | 0.30 cm | 0.17 cm | 0.80° |
| H3    | yes        | 1         | 0.24 cm | 0.12 cm | — |

Motion-based KF selection at frame 15 yielded ≤ 2 learnable keyframes for
all scenes (anchor at frame 0 plus K_refined others). Pose deltas are
sub-cm / sub-degree — well below the magnitude that could plausibly
explain a +2.63 cm aligned-ATE regression on H3, let alone a +5.34 cm
regression on H2.

## Failure mechanism

**The regressions are not caused by BR1's pose edits.** Pose deltas are
too small in magnitude. The 200-iteration joint optimization is changing
the **map**, and a differently-optimized map degrades tracking for the
remaining 85 frames after the BR1 event.

Concretely, what changes between baseline and BR1:

1. For 200 iterations at frame 15, Gaussians are co-optimized against a
   small K-frame window (K ∈ {2, 3}) with a pose parameter group present
   in the Adam optimizer. Even when the pose deltas stay small under the
   pose prior, the Gaussian gradients for that window are different from
   the normal mapping schedule — the loss landscape at frame 15 now
   balances photometric fit across K frames jointly rather than
   sequentially.
2. This re-sculpts the early Gaussian means/scales/opacities in ways the
   steady-state tracker wasn't optimized against.
3. The tracker's cosine-LR / early-stopping schedule is tuned for the
   normal map-growth trajectory. A "wrong" early map pushes the first
   post-bootstrap frames to converge at different poses, and velocity-
   init amplifies that delta forward.

The rigid-offset hypothesis from the ratio analysis — **that the 3.5–6.8x
raw/aligned ratio reflects a recoverable SE(3) bias in the early KFs** —
does not survive this test. If it were true, even tiny correct pose
deltas should move aligned ATE in the right direction. Moving it the
wrong direction means the ratio is not measuring a static bias that
bootstrap co-optimization can erase; it's measuring something the
pipeline keeps generating frame-by-frame that happens to average to a
near-SE(3) offset over 100 frames.

## Connection to prior rejections

This is the **seventh intervention** and continues the meta-pattern:
literature recipes have upstream dependencies our minimal baseline
lacks. Specifically:

- MonoGS bootstrap / global pose optimization presumes dense
  feature-based correspondences or ICP depth alignment between the
  refinement KFs — information independent of the photometric loss.
  We have only photometric+depth+SSIM across the same pixels the tracker
  already saw. Re-optimizing them at frame 15 doesn't introduce new
  geometric evidence; it redistributes the same evidence.
- Ratio-analysis intuition ("big raw/aligned ratio ⇒ rigid offset ⇒
  cheap to remove") is **wrong for photometric SLAM** where the raw
  trajectory is a Markov chain of velocity-init updates. The ratio is
  large because consecutive errors are correlated (velocity-init
  inherits yesterday's error), not because there's a static SE(3) bias
  waiting to be subtracted.

## Preserved side findings

- **Raw/aligned ratio does NOT imply extractable rigid bias.** Future
  interpretation of the ratio metric should treat it as a diagnosis of
  "errors are temporally correlated" not "errors factor as rigid offset
  + jitter". The ratio cannot be inverted into an intervention target.
- **Touching the bootstrap region destabilizes downstream tracking even
  with sub-cm pose deltas.** Any future intervention in [0, N_bootstrap]
  must be paired with a re-tracking step for all post-bootstrap frames,
  not just a pose writeback.
- **Motion-based KF selection keeps early KF count very low (K=1–2 at
  frame 15).** BR1's target surface was smaller than protocol assumed
  (which implicitly presumed ~5–7 KFs). Any future bootstrap-window
  intervention should either increase N_bootstrap until K ≥ 5 or force
  keyframes at fixed cadence during bootstrap.
- **H1 "no BR1 fired" + regression of +1.81 cm** is a pure run-to-run
  noise event. It is consistent with the ~±10 cm raw-ATE noise band
  observed in A2 (which translates to ~±2–3 cm aligned). Future BR
  variants should budget for this noise floor; a < 2 cm aligned
  improvement is indistinguishable from noise on BONN 100-frame stride=2.

## What stays in the codebase

- `SLAMPipeline._run_bootstrap_refinement` and `maybe_run_bootstrap` kept
  as dormant code gated by `slam.bootstrap.enabled` (default `false`).
- Config block `slam.bootstrap` documented in `configs/default.yaml`
  with REJECTED annotation so the next reader understands it shipped off.
- No changes to tracker, mapper, or Gaussian map. BR1 was strictly
  additive and fully reverts with `enabled: false`.

## Files produced

- `scripts/br1_eval.py` — evaluation harness (keep as template for any
  future bootstrap-window intervention).
- `research/experiments/bootstrap-refinement/protocol.md` — pre-reg.
- `research/experiments/bootstrap-refinement/analysis.md` — this doc.

## Pre-registration honored

Budgets were set on 2026-04-23 before running. No post-hoc threshold
tuning. Verdict was called exactly against the pre-registered decision
matrix.
