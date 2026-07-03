# LP1: Language-Driven Open-Vocabulary Dynamic Prior — Protocol

Registered 2026-07-04, before implementation results were known.

## Claim

Dynamic-object handling should not be limited to a closed detector
vocabulary. A text-defined dynamic vocabulary, scored against SAM+CLIP
segment features, can flag dynamic objects that YOLO's COCO classes
cannot (flagship: the BONN balloon), and fuses with detector evidence
through the existing per-Gaussian Bayesian belief field.

## Architecture

Three additions, no changes to the tracker or mapper:

1. **Sparse language evidence (extraction keyframes).** During language
   extraction the 768-D per-pixel CLIP feature map already exists.
   Score it against a configurable dynamic vocabulary (default:
   "a person", "an animal or pet", "a balloon or floating object",
   "an object being moved or carried") using the LangSplat contrastive
   relevancy vs canonical phrases; per-pixel max over prompts; pixels
   above `relevancy_thresh` form the language-dynamic mask. This mask
   (a) unions into the frame's dynamic mask and (b) drives
   `update_dynamic_belief` (occlusion-aware) with `belief_increase_lang`.

2. **Dense propagation (every frame).** Before tracking, Gaussians with
   `dynamic_belief > belief_mask_thresh` are projected into the predicted
   view and dilated, forming a belief-reprojection mask that unions into
   the frame's dynamic mask. Language evidence is sparse; the belief
   field carries it to every frame at negligible cost.

3. **Fusion.** YOLO (dense, closed-set) and language (sparse, open-set)
   evidence meet in the belief field and in the per-frame mask union.
   YOLO's vocabulary becomes one particular choice of dynamic
   vocabulary; the text prompts are runtime-configurable (capability
   demo E12 falls out for free).

CLIP scores appearance, not motion: the prior flags *movable/agent-like*
semantics. Geometric verification (existing instance depth-check for
non-person classes) remains the motion arbiter for ambiguous categories.

## Pre-registered evaluation

Scene h2 (BONN balloon), 100 frames stride 2, seeds 42/123/2024/7,
full mode, current code. The balloon is not a COCO class; YOLO masks
only the person.

| condition | expectation |
|---|---|
| C0 baseline (prior off) — fresh, post-9ed0af8 | establishes post-fix h2-full baseline |
| C1 prior on, default vocabulary | balloon pixels enter the dynamic mask; ATE improves or matches C0 |

Decision rule: accept if C1 mean ATE improves over C0 by > 1 SE with no
h1/h3 regression (h1 spot-check 2 seeds; h3 control 1 seed — expect
language-dynamic coverage ~0% on Replica: nothing movable present).
Secondary metric: fraction of GT balloon pixels covered by the dynamic
mask (qualitative filmstrip for the paper figure).

## Knobs (fixed before evaluation; not tuned on the eval seeds)

- relevancy_thresh: 0.6
- belief_increase_lang: 0.5
- belief_mask_thresh: 0.5
- belief mask dilation: 15 px
