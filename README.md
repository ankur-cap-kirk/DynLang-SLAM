# DynLang-SLAM

**Open-vocabulary dynamic-scene SLAM with 3D Gaussian Splatting.**

DynLang-SLAM builds a language-queryable 3D Gaussian map from an RGB-D stream while
detecting and masking dynamic objects, so moving people and objects neither corrupt
camera tracking nor pollute the semantic map. Text queries ("monitor", "chair") localize
objects in the reconstructed 3D scene; queries for dynamic objects that were masked out
("person") correctly return nothing.

> Research code under active development — a paper is in preparation. Expect rough edges;
> issues and questions are welcome.

## Highlights

- **Dynamic-object handling**: per-instance detection (YOLOv8-seg) with instance-level
  depth verification, temporal filtering, occlusion-aware Bayesian per-Gaussian dynamic
  belief, and contamination cleanup.
- **Language-embedded map**: CLIP ViT-L/14 features distilled through SAM2 segments into
  a compact 16-D per-Gaussian latent (online-trained autoencoder), seeded directly from
  observations at Gaussian creation.
- **Dynamic-aware losses**: RGB, depth, *and SSIM* losses all respect the dynamic mask
  (an unmasked SSIM term silently negates dynamic masking — see commit history).
- **Runs on one consumer GPU**: all results below were produced on a single 8 GB RTX 5070
  laptop GPU (peak usage ≈ 1.4 GB for BONN scenes).

## Results

ATE RMSE (cm), mean ± std over **8 seeds** per cell, 100 frames per sequence:

| Sequence | No masking | + Dynamic masking | + Language (full) |
|---|---|---|---|
| BONN person_tracking | 12.78 ± 2.20 | **8.64 ± 1.53** | 10.59 ± 1.86 |
| BONN balloon | 13.45 ± 2.43 | 10.48 ± 1.59 | **10.14 ± 1.47** |
| Replica room0 (static control) | 5.46 ± 2.19 | 6.58 ± 0.87 | 6.71 ± 1.44 |

Raw per-run records: [`results/sweep/sweep_results.jsonl`](results/sweep/sweep_results.jsonl),
reproducible via `python scripts/sweep_multiseed.py --drive`.

## Installation

Requires Python ≥ 3.10, CUDA-capable GPU (8 GB VRAM sufficient), PyTorch with CUDA.

```bash
git clone https://github.com/ankur-cap-kirk/DynLang-SLAM.git
cd DynLang-SLAM
pip install torch torchvision  # match your CUDA version, see pytorch.org
pip install ultralytics open_clip_torch numpy opencv-python
pip install "sam2 @ git+https://github.com/facebookresearch/sam2.git"
```

Download the SAM2 checkpoint into `checkpoints/`:

```bash
# sam2.1_hiera_tiny.pt from https://github.com/facebookresearch/sam2
```

A pinned `requirements.txt` is planned alongside the paper release.

## Datasets

Datasets are **not** redistributed here — download them from their owners and respect
their licenses (Replica is research-only; please cite BONN/TUM/Replica when using them):

- **BONN RGB-D Dynamic**: https://www.ipb.uni-bonn.de/data/rgbd-dynamic-dataset/ →
  `data/BONN/<sequence>/`
- **TUM RGB-D**: https://cvg.cit.tum.de/data/datasets/rgbd-dataset → `data/TUM/...`
- **Replica**: `python scripts/download_replica.py` → `data/Replica/<scene>/`

## Live demo: query a 3D map with text

After any run that saves a checkpoint (e.g. `scripts/eval_miou_replica.py`),
open the map in an interactive [Rerun](https://rerun.io) viewer and type
free-text queries — matching regions light up in place:

```bash
pip install rerun-sdk matplotlib
python scripts/demo_live_query.py --checkpoint results/sweep/miou_slam_checkpoint.pt
```

Type `sofa`, `lamp`, `door` in the terminal; each query becomes a step on
the Rerun timeline you can scrub back through. On a dynamic-scene map
(BONN), querying `person` returns scattered noise rather than a compact
blob — the person was masked out during mapping and is *not in the map*
(measured: its relevancy matches the noise floor of a scene where no
person ever existed).

Flags: `--save demo.rrd --queries "sofa,lamp"` records a replayable file
headless; `--up z-up` if your map's world convention leaves the scene
tilted. A presenter's script lives in `report/demo_runbook.md`.

## Quickstart

```bash
# End-to-end test on BONN person_tracking (static vs dynamic masking comparison)
python scripts/test_bonn_slam.py

# Multi-seed statistical sweep (3 scenes x 3 modes x 8 seeds, resumable)
python scripts/sweep_multiseed.py --drive
python scripts/sweep_multiseed.py --summarize
```

Configuration lives in `configs/default.yaml` (dynamic masking, language pipeline,
tracker, and mapper settings are all documented inline).

## License

This project's code is licensed under the **Apache License 2.0** (see [LICENSE](LICENSE)).

Third-party components have their own licenses:

| Component | License | Note |
|---|---|---|
| [ultralytics](https://github.com/ultralytics/ultralytics) (YOLOv8) | **AGPL-3.0** | Optional dependency for the dynamic detector. Using this path may impose AGPL obligations on derived works — consult the ultralytics license. |
| [SAM2](https://github.com/facebookresearch/sam2) | Apache-2.0 | |
| [open_clip](https://github.com/mlfoundations/open_clip) | MIT-style | CLIP ViT-L/14 weights: OpenAI |
| [CoTracker](https://github.com/facebookresearch/co-tracker) | CC BY-NC 4.0 | **Disabled by default** (`dynamic.pips.enabled: false`); non-commercial license applies only if you enable it. |

Datasets (BONN, TUM, Replica) are subject to their own terms and are not distributed
with this repository.

## Citation

A paper is in preparation. Until then, please cite this repository:

```bibtex
@misc{dynlangslam2026,
  title  = {DynLang-SLAM: Open-Vocabulary Dynamic-Scene Gaussian Splatting SLAM},
  author = {Guruprasad, Ankur},
  year   = {2026},
  url    = {https://github.com/ankur-cap-kirk/DynLang-SLAM}
}
```

## Acknowledgments

Builds on ideas from SplaTAM, LangSplat, DG-SLAM, BDGS-SLAM, WildGS-SLAM, and SGS-SLAM,
and on 3D Gaussian Splatting (Kerbl et al., 2023).
