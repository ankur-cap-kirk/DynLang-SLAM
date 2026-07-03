# Premise Check: Is the Balloon an ATE Problem? (2026-07-04)

## Question

LP1's flagship framing assumed the unmasked balloon materially hurts h2
tracking. Test with an oracle: DG-SLAM depth-warp motion mask fed
GROUND-TRUTH poses (perfect camera-motion compensation; injected via
the new `external_dynamic_mask_fn` pipeline hook), threshold 0.3 m,
4-frame intersection window, 7 px dilation.

## Result (h2, 100 frames, full mode, seeds 42/123/2024/7)

| condition | ATE (cm) | extra mask |
|---|---|---|
| C0 person-only (banked) | 9.95 ± 1.77 | — |
| LP1 v1 (banked) | 11.51 ± 1.67 | ~5% |
| GT-pose motion oracle | **12.27 ± 1.09** | 8.9% |

## Verdict

**Premise rejected.** Even oracle-quality class-agnostic motion masking
loses to person-only masking on h2. Masking beyond the person removes
more static tracking signal than the contamination it prevents; the
balloon (small, fast, motion-blurred) contributes little pose error.
This retroactively explains two prior rejections in this repo: the PIPs
mask "breaking H2" and the unadopted depth-warp branch.

## Reframed contribution (supersedes the ATE-improvement framing)

1. **ATE (h1)**: masking gains live on the person. Next experiment:
   language-only ("a person" prompt, YOLO off) vs YOLO-only on
   person_tracking — if equal, the open-vocabulary prior SUBSUMES the
   closed detector.
2. **Map purity (h2)**: the balloon's harm is to the MAP (ghost
   Gaussians, render artifacts, spurious query hits), not the pose.
   Metric: query "balloon" on C0 vs C1 maps against the person-free
   noise floor (negative-control machinery); plus rendering comparison
   in balloon-crossed regions.
3. **Capability**: runtime text-defined dynamic vocabulary (unchanged).

Raw records: `results/sweep/exp_oracle_premise.jsonl`.
