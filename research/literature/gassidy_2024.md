# Gassidy — Gaussian Splatting SLAM in Dynamic Environments (2024)

Paper: https://arxiv.org/abs/2411.15476

## Core idea: per-iteration loss flow + GMM
- Run mapping optimization for 40 iters, record **loss trajectory ΔL_k = L_k - L_{k+1}** per object (per instance-segmented region) across iters.
- Dynamic objects: "higher and more fluctuating" loss values.
- Static: loss decreases monotonically.
- **GMM classifier** on ΔL sequence → P(object is dynamic).
- Threshold: **99.9%** posterior.

## Dependencies
- Upstream: **YOLO instance segmentation** (required; not class-agnostic in the strict sense, though they claim reduced reliance on class types).
- Loss adaptive weight: `L = λ_a · L_pho + (1 - λ_a) · L_geo`, `λ_a` depends on depth quality.

## BONN numbers
- balloon: **2.6 ± 0.8 cm**
- person_tracking (ps_track): **10.3 ± 4.4 cm**

## Relevance to DynLang-SLAM
- Interesting signal: **optimization-time loss behavior** can separate dynamic from static.
- Requires instance grouping upstream (YOLO) → still fails on balloon unless YOLO fires.
- Moderate complexity (per-object loss tracking, GMM fit per frame) — maybe week 2 candidate.
- The 2.6 cm balloon number is a strong target; their YOLO must pick up balloon somehow (perhaps "sports ball" class or looser threshold than our 0.5).
