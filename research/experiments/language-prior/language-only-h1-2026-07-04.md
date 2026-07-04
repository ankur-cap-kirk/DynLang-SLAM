# Language-Only Dynamic Handling on h1 (2026-07-04)

## Question

Can the text prior alone (YOLO neutered, `dynamic_classes=[]`) match the
closed detector on person_tracking? If yes, the open-vocabulary prior
*subsumes* YOLO.

## Result (h1, 100 frames, full mode, seeds 42/123/2024/7)

| seed | ATE (cm) | avg mask | belief-hot Gaussians |
|------|----------|----------|----------------------|
| 42   | 7.23     | 1.1%     | 1275 |
| 123  | 8.35     | 1.9%     | 1994 |
| 2024 | 11.97    | 0.6%     | 0    |
| 7    | 11.86    | 0.2%     | 7    |

Mean 9.85 ± 2.10.
References: YOLO-only 8.47 ± 1.20 (n=8); no masking 12.78 ± 2.20 (n=8).

## Interpretation: bimodal, not mediocre

Two regimes, split by whether the belief field bootstrapped:

- **Caught (42, 123):** 1000+ hot Gaussians, ATE 7.23/8.35 — BEATS the
  YOLO-only mean, while masking ~1-2% of the frame vs YOLO's ~11%.
  The belief-reprojection carries sparse language evidence densely and
  precisely.
- **Cold-start failure (2024, 7):** ~0 hot Gaussians, ATE 11.97/11.86 —
  collapses to no-masking. With YOLO off there is no fallback, so if
  early extraction keyframes don't deposit enough belief on the person
  (relevancy_thresh / belief_mask_thresh too high, AE-freeze timing, or
  the person not prominently framed early), the field never crosses
  threshold and nothing is ever masked.

## Verdict

Ceiling is ABOVE YOLO (2/4 seeds beat it with a far thriftier mask), but
reliability is gated by a cold-start belief-bootstrap problem. Not a
subsumption claim yet; a partial success with an identified, fixable
failure mode.

## Next

Cold-start fixes to try (h1, same 4 seeds):
1. Lower `belief_mask_thresh` / `relevancy_thresh` so the person catches
   with less accumulated evidence.
2. Earlier + denser language extraction during warmup (extract_every_n
   low, or force an extraction in the first few keyframes).
3. Larger `belief_increase` per language hit.
Target: reduce the two cold-start collapses; if mean -> ~8 with tighter
std, the "subsumes YOLO" claim becomes real.

Raw records: `results/sweep/exp_langonly.jsonl`.
