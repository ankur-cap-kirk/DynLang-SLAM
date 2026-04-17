# Experiment: Silhouette gate on tracker loss (intervention C)

## Motivation
Literature review (see `research/to_human/sota_gap_closing_methods.html`)
identified 3 architectural interventions likely to dent the SOTA gap. This is
the cheapest to implement — starting here to (a) confirm our tooling still
works end-to-end and (b) quantify one independent contribution before
stacking.

## What the change does
During **tracking only**, multiply the photometric / depth / SSIM residuals by
a hard binary mask `(alpha_rendered > 0.95)`. Pixels whose rendered alpha falls
below the threshold do not contribute to the pose gradient.

- **Mapping is unchanged** — mapping needs loss on uncovered regions so it can
  grow into them.
- **Depth validity** also raised from `alpha > 0.5` to `alpha > 0.95` in
  tracking mode.
- **SSIM** computed after multiplying both rendered and GT RGB by the gate;
  regions outside the gate become `(0, 0)` which yield SSIM=1 and contribute 0
  to the `1-SSIM` loss.

## Citation
SplaTAM (Keetha et al., CVPR 2024) uses `S(p) > 0.99`. We relax to 0.95
because our iteration budget (50) is larger than SplaTAM-S (10) and smaller
than full SplaTAM (40), and our map density during early frames is lower.

## Hypothesis
- H1: On BONN person_tracking (baseline 20.32 cm full mode), alpha gating
  reduces ATE by 1–3 cm by preventing the tracker from fitting un-mapped
  regions during the discrete drift event at frame ~65.
- H2: On BONN balloon (baseline 57.35 cm full mode), effect is smaller because
  the balloon failure is not about coverage — it's about YOLO missing the
  balloon class. Gate won't help there.
- H0 (accept-criterion): If person_tracking full ATE does not improve by at
  least 0.5 cm, we reject this change and move on to intervention B
  (depth-warp mask).

## Code diff summary
- `dynlang_slam/core/losses.py`: added `tracking_mode` + `alpha_gate_thresh`
  kwargs to `compute_losses`. When tracking_mode: hard gate instead of soft
  alpha weighting; stricter depth-validity threshold; SSIM over gated images.
- `dynlang_slam/slam/tracker.py`: added `alpha_gate_thresh` + `use_alpha_gate`
  to `Tracker.__init__`, passed through to `compute_losses`.
- `dynlang_slam/slam/pipeline.py`: wired config → Tracker constructor.
- `configs/default.yaml`: added `slam.tracking.use_alpha_gate: true` and
  `slam.tracking.alpha_gate_thresh: 0.95`.

## Protocol
1. Run `scripts/test_bonn_full.py` on person_tracking (100 frames, stride 2),
   log to `person_tracking_log.txt`.
2. If H1 passes, also run on balloon (edit SEQUENCE pointer), log to
   `balloon_log.txt`.
3. Record baseline ATE (from prior experiments):
   - person_tracking full: 20.32 cm
   - balloon full: 57.35 cm

## Expected runtime
~5-10 min per sequence x 2 sequences = ~20 min total.

## Accept criteria (pre-registered)
| Sequence        | Baseline | Accept if full mode ATE ≤ |
|-----------------|---------:|--------------------------:|
| person_tracking |  20.32   |             19.82 (-0.5)  |
| balloon         |  57.35   |             56.85 (-0.5)  |

If person_tracking improves by ≥ 0.5 cm AND balloon does not regress by more
than 1 cm, accept and commit to main. Otherwise, revert and document why.
