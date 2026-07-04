# Language-Only Cold-Start Campaign (2026-07-04)

Four operating points of the threshold/decay space on h1 language-only
(YOLO neutered), 100 frames, seeds 2024/7/42/123 (failure seeds first).
References: YOLO-only 8.47 ± 1.20 (n=8); no masking 12.78 ± 2.20 (n=8).

| variant | knobs | ATE (4 seeds) | belief coverage | failure mode |
|---|---|---|---|---|
| baseline | thresh .6, decay .07 | 9.85 ± 2.10 (7.23/8.35/11.97/11.86) | 0-10% | decay race → bimodal: catches beat YOLO, misses collapse |
| CS1 | mask_thresh .3, incr .7 | 12.15 (stopped, n=1) | 37% | single-hit trust → furniture floods mask |
| CS2 | decay .01 | 10.36 ± 0.36 | ~28% | noise persists → uniform over-mask |
| CS3 | CS2 + 3 prompts, thresh .7 | 9.99 ± 1.31 (8.25-11.82) | ~2% | person under-caught → uniform under-mask |

## Conclusion

No fixed relevancy threshold separates person from furniture reliably
with single-'whole'-scale CLIP features: the operating curve swings
from under- to over-masking with no dependable middle. BUT the baseline
regime proved the ceiling: when belief catches (seeds 42/123), the
language-only system BEATS the YOLO mean (7.23/8.35 vs 8.47) while
masking ~1-2% of the frame vs YOLO's ~11%.

## v2 design (justified by this table)

**Language proposes, geometry confirms.** Generous relevancy threshold
(recall-oriented, e.g. 0.55-0.6) to guarantee the person is proposed;
connected components of the language mask are treated as instances and
must pass the existing instance-level depth-motion verification (the
same machinery already gating non-person YOLO classes) before entering
mask/belief. Static furniture proposals are vetoed by physics, so the
threshold no longer has to do precision's job. Persistence (slow decay)
then amplifies only geometry-confirmed evidence.

Raw records: `results/sweep/exp_langonly.jsonl` (experiments
lang_only_h1, _cs1, _cs2, _cs3).
