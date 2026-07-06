# DynLang-SLAM — Professor Showcase Runbook

## Setup (2 minutes, do before the meeting)

```bash
cd C:\Users\ankur\Desktop\research
# Live viewer on the Replica room0 map:
python DynLang-SLAM\scripts\demo_live_query.py
# Live viewer on the BONN dynamic-scene map:
python DynLang-SLAM\scripts\demo_live_query.py --checkpoint DynLang-SLAM\results\sweep\demo_bonn_checkpoint.pt
```

A Rerun 3D window opens with the reconstructed scene. Type queries in the
terminal; the map recolors in place. Fallback if anything misbehaves:
pre-recorded replays (open with `rerun <file>`):
`results/demo_room0.rrd`, `results/demo_bonn.rrd`.

## Demo flow (5 minutes)

1. **Room0 (static scene).** Rotate the map. Type `sofa`, then `lamp`,
   then `door` — each lights up in its true 3D location. Point out: the
   system never saw a semantic label; classification is zero-shot from
   the text, through CLIP features distilled into 16 numbers per Gaussian.
2. **BONN (dynamic scene).** Explain: a person walked through every frame
   of this recording. Type `monitor`, `chair`, `screen` — each lights a
   compact blob (spread ~0.7-0.9 m). Then type `person` — the heat is
   scattered speckle across the whole room (spread ~1.8 m), because the
   person is NOT in the map: the system detected, masked, and excluded
   them during mapping. The contrast to show is coherence, not darkness:
   CLIP similarity never returns zero for anything, so "absent" looks
   like noise, and it measurably matches the noise floor of a scene where
   no person ever existed (0.63 vs 0.60 mean relevancy, decoder route).
3. **The trajectory numbers.** April: 22.3 cm ATE on this sequence.
   Today: 8.5 ± 1.2 cm over 8 seeds — from five root-caused fixes, each
   documented in `research/experiments/`.

## Key numbers (if asked)

| claim | number |
|---|---|
| Tracking, BONN person_tracking | 8.47 ± 1.20 cm (n=8); was 22.3 in April |
| Dynamic masking benefit | 12.78 -> 8.64 cm (~2 sigma, n=8) |
| Language costs nothing | full 8.47 vs dynamic-only 8.64 (n=8 each) |
| Negative control | masked person 0.220 ~= never-existed 0.241 |
| Zero-shot 3D mIoU (no semantic supervision) | 0.066-0.071, room0, 20 classes |
| Language-only masking ceiling | beats YOLO on catching seeds (7.23/8.35 vs 8.47) with ~1-2% masked vs 11% |
| Hardware | everything on one 8 GB laptop GPU |

## Story arc (if asked "what's the contribution?")

Open-vocabulary dynamic SLAM: the dynamic vocabulary is *text*, not a
fixed detector class list. Current evidence: language-only masking beats
YOLO when the belief field catches (reliability work in progress —
"language proposes, geometry confirms" v2 under evaluation); dynamic
objects are provably absent from the queryable map (negative control);
all claims carry n=8 error bars and pre-registered accept/reject
protocols, including the honest rejections.
