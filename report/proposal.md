**1) Title**

DynLang-SLAM: Dynamic-Aware Open-Vocabulary Language-Embedded 3D Gaussian Splatting SLAM

**2) Students on the team**

Ankur Guruprasad (solo)

**3) What is the problem you are solving?**

Current 3D Gaussian Splatting (3DGS) based SLAM systems produce high-quality geometric reconstructions but suffer from two critical limitations: (a) they generate purely geometric maps with no semantic understanding, meaning a robot cannot answer natural language queries like "where is the coffee mug?" from the reconstructed map, and (b) they assume a fully static environment, which causes catastrophic tracking failures when dynamic objects such as people or pets are present. These two limitations severely restrict the deployment of 3DGS-SLAM in real-world robotics and augmented reality applications where scenes are rarely static and semantic understanding is essential.

**4) What is the "new" or "novel" aspect? Why has this not been done before?**

While SplaTAM (CVPR 2024) demonstrated 3DGS for dense SLAM, LangSplat (CVPR 2024) showed language features can be embedded into Gaussians, and DGS-SLAM (2024) addressed dynamic masking, no existing system unifies all three capabilities into a single framework. DynLang-SLAM is novel because it integrates: (1) real-time 3DGS-based tracking and mapping, (2) per-Gaussian CLIP language features compressed via a learned autoencoder (768D to 16D) for open-vocabulary 3D queries, and (3) YOLOv8-Seg dynamic object masking with temporal consistency filtering and Gaussian contamination cleanup. This combination has not been attempted because each component introduces competing demands on GPU memory and compute -- language feature rendering is expensive (LangSplat achieves only 8.2 FPS even on A100 GPUs), dynamic masking adds inference overhead, and all must run alongside the already demanding Gaussian rasterization and pose optimization loop. Our approach addresses this through compact feature compression, selective per-keyframe language extraction, and efficient alpha-masked loss computation that excludes both dynamic and unmapped regions.

**5) What are some preliminary datasets, code bases you will start from for your experiments?**

- **Dataset:** Replica dataset (8 indoor scenes, 2000 frames each, ground-truth RGB-D and camera poses)
- **Code bases:** We are building our own implementation (DynLang-SLAM) from scratch using:
  - gsplat (Nerfstudio's CUDA-accelerated Gaussian rasterizer) for differentiable rendering
  - OpenCLIP (ViT-L/14) for language feature extraction
  - SAM (Segment Anything Model) for multi-scale object masks
  - YOLOv8x-Seg (Ultralytics) for dynamic object detection
  - PyTorch 2.10 with CUDA 12.8
- **Baselines for comparison:** SplaTAM, MonoGS (tracking/reconstruction), LangSplat (language grounding)
- **Evaluation metrics:** ATE RMSE (tracking accuracy), PSNR/SSIM/Depth L1 (rendering quality), precision/recall (language queries)

**6) Tentative timeline till the end of the semester**

- **Week 1 (Completed):** Core SLAM pipeline -- implemented tracking (6-DOF pose optimization via photometric + depth loss) and mapping (Gaussian parameter optimization + silhouette-based expansion) with gsplat CUDA renderer at full resolution (680x1200). Current results: sub-1cm ATE on 4/8 Replica scenes.
- **Week 2:** Language pipeline -- integrate CLIP feature extraction, train scene-specific autoencoder (768D to 16D), attach per-Gaussian language features using SAM multi-scale masks, implement open-vocabulary text query interface.
- **Week 3:** Dynamic object masking -- integrate YOLOv8-Seg for instance segmentation, implement alpha-masked loss excluding dynamic regions from tracking/mapping, add temporal consistency filter across sliding window, implement Gaussian contamination cleanup pass.
- **Week 4:** Full evaluation and ablation study across all 8 Replica scenes. Measure tracking accuracy (ATE), rendering quality (PSNR/SSIM), language grounding accuracy, and ablate each component. Write final project report.
