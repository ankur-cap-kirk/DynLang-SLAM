# Analysis: Opacity reliability mask + loss-weight flip (intervention D1)

## Result summary — REJECTED

All three variants fail H1 decisively. Pre-registered accept criterion:
person_tracking full ATE ≤ **14.00 cm**; reject budget ≤ **22.32 cm**.
Baseline 20.32.

| Variant                       | Static | Dynamic | Full   | Full Δ   | Verdict |
|-------------------------------|-------:|--------:|-------:|---------:|:--------|
| baseline                      |  28.36 |   21.85 |  20.32 |    —     | —       |
| D1 combined (mask + weights)  |  59.54 |   47.17 |  52.20 | **+31.88** | FAIL   |
| D1a opacity-only (old weights)|  20.49 |   21.35 |  28.99 | **+8.67**  | FAIL   |
| D1b weights-only (soft mask)  |  53.44 |   35.45 |  40.91 | **+20.59** | FAIL   |

H2 (balloon) and H3 (Replica room0) **not run** — H1 primary fails
decisively across three variants, no point spending GPU budget.

## Decomposition diagnosis

We ran D1 as the combination of two independent changes — hard alpha
reliability mask (opacity) + DG-SLAM loss weights (rgb=0.9, depth=0.1)
— and when the combined run regressed +31.88 cm, split them to
identify the culprit:

### D1b (weights-only) — depth anchor removal breaks tracking

Flipping base weights from `rgb=0.5, depth=1.2` to `rgb=0.9, depth=0.1`
regresses +20.59 cm on its own. Root cause is visible in tracker.py:

```python
# tracker.py lines 161-170:
depth_scale = 1.0              # constant, never reduce depth anchor
rgb_scale = 0.5 + 0.5 * t     # 0.5 -> 1.0
ssim_scale = 0.5 + 0.5 * t    # 0.5 -> 1.0
```

The tracker has a **time-varying weight schedule specifically tuned
for depth=1.2**. Its comment `"never reduce depth anchor"` encodes the
design assumption that depth is the convex-basin signal that keeps
Adam out of photometric local minima. With base depth=0.1, the
"anchor" is roughly 1% of RGB throughout — no longer an anchor.

DG-SLAM's 0.9/0.1 weights work because DG-SLAM runs this as a
**refinement** stage after DROID-VO provides the coarse pose. Our
tracker does the full job from velocity init, which requires a much
stronger depth prior to converge.

### D1a (opacity-only) — hard RGB mask causes densification explosion

Keeping old weights but switching RGB loss mask from soft alpha
weighting to hard threshold (`alpha > 0.5`, matching what depth was
already using) produces a more nuanced result:

- **Static mode improves −7.87 cm** (28.36 → 20.49) — genuine signal.
- **Full mode regresses +8.67 cm** (20.32 → 28.99).

The mode-dependent behavior points to a side effect, not a direct
tracking failure. **Gaussian count**:

| Mode    | Baseline | D1a    | Ratio |
|---------|---------:|-------:|------:|
| Static  |  20594   | 44093  | 2.14× |
| Dynamic |  19430   | 29922  | 1.54× |
| Full    |  21547   | 36232  | 1.68× |

Hard RGB mask changes which pixels the mapper considers "well-
represented," which flows into the densification decision. With the
old soft weighting, low-alpha pixels contributed proportionally but
were not flagged as under-represented. With the hard threshold,
anything below 0.5 is effectively invisible to the RGB loss —
densification sees those regions as needing more Gaussians. Map
explodes.

The Full-mode regression is likely driven by this side effect: larger
map + language pipeline load = more param drift, longer convergence,
more local minima. Not a property of the mask design but of the
densification coupling.

### Combined D1 — effects compound

D1 (+31.88) is strictly worse than D1b alone (+20.59) which is
strictly worse than D1a alone (+8.67). Ordering confirms the
weight flip is dominant and the mask change adds a second hit via
the densification-coupling channel.

## Structural mismatch with DG-SLAM

| What DG-SLAM assumes        | What we have                    |
|-----------------------------|---------------------------------|
| DROID-VO coarse pose        | velocity init                   |
| Photometric = refinement    | Photometric = full tracking     |
| Hard alpha mask on stable map | Hard alpha mask on growing map |
| Dense Gaussian map (>100k)  | Sparse map (16–40k)             |
| Weight schedule retuned     | Legacy schedule for old weights |

Every single assumption that makes DG-SLAM's 0.9/0.1 weights work
is absent in our regime. This is the **fifth rejected intervention
with the same root pattern**: literature recipe assumed a better
upstream or complementary component than we have (C: SplaTAM 200k
init; B: DG-SLAM DROID-VO; B': dense GT tracks; A1: MonoGS ICP +
large KF graph; D1: DG-SLAM DROID-VO + retuned schedule).

## Interesting side findings (preserved even though D1 rejects)

1. **Hard RGB mask changes densification behavior ~2×.** This is
   non-obvious — the loss function does not directly touch the
   densification code, but the gradient-magnitude signal used for
   densify decisions is affected by which pixels contribute. Worth
   remembering for future interventions.

2. **Static-mode improvement is real** (−7.87 cm with D1a). On
   clean single-scene rendering the hard mask does help. If we
   ever revisit this in a regime where map growth is already
   constrained (e.g., post-densification-cap), the hard mask on
   its own may be worth another look.

3. **Our time-varying weight schedule is load-bearing.** The
   `depth_scale = 1.0 # constant` line in tracker.py is not a
   cosmetic detail — it's the primary reason tracker doesn't
   diverge on hard frames. Any future loss-weight intervention
   must account for (or redesign) this schedule, not just flip
   the base weights.

## Right mechanism for the gap we still have

After C/B/B'/A1/D1, the failure mode identified in research-state.yaml
remains: **discrete single-frame catastrophic drift event at frame
60-70**. Every intervention we've tried so far is either:

- **Mapping-side** (C, B, B') — runs AFTER the tracker commits a bad
  pose, can't undo the drift event.
- **Cross-keyframe** (A1) — redistributes error but can't correct a
  single bad pose without a stronger anchor.
- **Loss-formulation** (D1) — sensitive to unrelated design coupling.

The remaining interventions **A2 (tracking-failure retry / multi-
hypothesis init)** and **B3 (Tukey robust loss in tracker)** are
tracker-side single-frame interventions. They don't share any of
these failure modes:

- **Single-frame action**, not windowed.
- **Direct tracker fix**, not mapping-side.
- **No gauge / map coupling**.
- **No upstream component assumed**.
- **Directly addresses the drift-event failure mode.**

A2 is the next natural step: when tracker loss indicates a hard
frame, try multiple pose initializations and keep the best. This
is the poor-man's substitute for DG-SLAM's DROID-VO frontend —
instead of a learned coarse estimate, use a small search over
plausible initializations.

## Disposition

- Revert config:
  - `loss.rgb_weight: 0.5` (was 0.9)
  - `loss.depth_weight: 1.2` (was 0.1)
  - `loss.use_hard_rgb_mask: false` (was true via default)
- Leave the `use_hard_rgb_mask` / `reliability_thresh` knobs in
  place (zero cost to keep, may be useful for future ablations or
  revisiting under different densification-cap conditions).
- Branch merged to main with `--no-ff`.
- `research-state.yaml`: D1 rejected, next = A2 retry.

## Budget note

Intervention #5 of the project, #5 rejected. Pattern across C, B,
B', A1, D1 is now overwhelming evidence that **architectural
interventions coupling the tracker's loss or the map's optimization
to the rest of the pipeline hit non-local coupling issues in our
regime**. The remaining interventions (A2 retry, B3 Tukey) are
explicitly tracker-side, single-frame, locally-scoped — they do
not have this failure mode.

Each rejected intervention has nevertheless produced genuine
diagnostic value:
- C revealed sparse-map sensitivity to hard gates
- B revealed velocity-init's depth-residual dominance
- B' revealed mask-after-tracking can't fix tracker drift
- A1 revealed ICP-anchor dependency in local BA
- D1 revealed weight-schedule coupling and RGB-mask→densification coupling

Writeup-wise, the five rejections tell a coherent story: literature
recipes have hidden component dependencies our baseline doesn't
satisfy, and the drift event requires a tracker-side fix.
