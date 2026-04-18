"""SLAM pipeline for DynLang-SLAM.

Orchestrates the tracking-mapping loop with keyframe management
and optional language feature extraction (CLIP + SAM + autoencoder).
"""

import time
from collections import deque

import torch
import numpy as np

from ..core.gaussians import GaussianMap
from ..core.renderer import GaussianRenderer
from .tracker import Tracker
from .mapper import Mapper
from ..utils.camera import fast_se3_inverse


class SLAMPipeline:
    """Main SLAM pipeline: alternates between tracking and mapping.

    Architecture:
        For each frame:
            1. Track: optimize camera pose against current Gaussian map
            2. Check keyframe criteria
            3. If keyframe: Map (optimize Gaussians + add new ones)
    """

    def __init__(self, cfg, intrinsics: dict, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self.intrinsics = intrinsics

        K = intrinsics["K"].to(device)
        self.K = K
        self.width = intrinsics["width"]
        self.height = intrinsics["height"]

        # Renderer
        self.renderer = GaussianRenderer(near=0.01, far=100.0)

        # Render downscale for optimization speed
        render_downscale = getattr(cfg.slam, 'render_downscale', 4)

        # Tracker (with coarse-to-fine from GS-SLAM)
        self.tracker = Tracker(
            renderer=self.renderer,
            num_iterations=cfg.slam.tracking.iterations,
            lr_pose=cfg.slam.tracking.lr_pose,
            loss_weights={
                "rgb_weight": cfg.loss.rgb_weight,
                "depth_weight": cfg.loss.depth_weight,
                "ssim_weight": cfg.loss.ssim_weight,
            },
            device=device,
            render_downscale=render_downscale,
            lr_end_factor=getattr(cfg.slam.tracking, 'lr_end_factor', 0.05),
            coarse_to_fine=getattr(cfg.slam.tracking, 'coarse_to_fine', True),
            coarse_downscale=getattr(cfg.slam.tracking, 'coarse_downscale', 4),
            coarse_ratio=getattr(cfg.slam.tracking, 'coarse_ratio', 0.6),
            use_soft_dynamic=getattr(cfg.dynamic, 'enabled', False) and getattr(cfg.dynamic, 'use_soft_weights', True),
            early_stop_patience=getattr(cfg.slam.tracking, 'early_stop_patience', 8),
            early_stop_rel_thresh=getattr(cfg.slam.tracking, 'early_stop_rel_thresh', 0.001),
            use_hard_rgb_mask=getattr(cfg.loss, 'use_hard_rgb_mask', True),
            reliability_thresh=getattr(cfg.loss, 'reliability_thresh', 0.5),
        )

        # Mapper
        self.mapper = Mapper(
            renderer=self.renderer,
            num_iterations=cfg.slam.mapping.iterations,
            lr_means=cfg.slam.mapping.lr_means,
            lr_scales=cfg.slam.mapping.lr_scales,
            lr_quats=cfg.slam.mapping.lr_quats,
            lr_opacities=cfg.slam.mapping.lr_opacities,
            lr_colors=cfg.slam.mapping.lr_colors,
            lr_lang_feats=cfg.slam.mapping.lr_lang_feats,
            loss_weights={
                "rgb_weight": cfg.loss.rgb_weight,
                "depth_weight": cfg.loss.depth_weight,
                "ssim_weight": cfg.loss.ssim_weight,
            },
            new_gaussian_thresh=cfg.slam.mapping.new_gaussian_thresh,
            device=device,
            render_downscale=render_downscale,
            lr_end_factor=getattr(cfg.slam.mapping, 'lr_end_factor', 0.1),
            densify_grad_thresh=getattr(cfg.gaussians, 'densify_grad_thresh', 0.0002),
            densify_interval=getattr(cfg.gaussians, 'densify_interval', 5),
            densify_max_scale=getattr(cfg.gaussians, 'densify_max_scale', 0.05),
            use_soft_dynamic=getattr(cfg.dynamic, 'enabled', False) and getattr(cfg.dynamic, 'use_soft_weights', True),
            refine_poses=getattr(cfg.slam.mapping, 'refine_poses', False),
            lr_pose_trans=getattr(cfg.slam.mapping, 'lr_pose_trans', 1.0e-4),
            lr_pose_quat=getattr(cfg.slam.mapping, 'lr_pose_quat', 5.0e-4),
            pose_prior_weight=getattr(cfg.slam.mapping, 'pose_prior_weight', 10.0),
            use_hard_rgb_mask=getattr(cfg.loss, 'use_hard_rgb_mask', True),
            reliability_thresh=getattr(cfg.loss, 'reliability_thresh', 0.5),
        )

        # Keyframe management
        self.keyframe_every_n = cfg.slam.keyframe.every_n_frames
        self.max_keyframes = cfg.slam.keyframe.max_keyframes
        self.keyframes: list[dict] = []  # list of {frame, pose}
        self.keyframe_window_size = 5  # recent keyframes used for mapping

        # State
        self.estimated_poses: list[torch.Tensor] = []
        self.frame_count = 0
        self._keyframe_count = 0  # monotonic keyframe counter (for language cadence)
        self._recent_losses = deque(maxlen=20)  # for outlier rejection
        self._prev_depth: torch.Tensor | None = None

        # Language pipeline (lazy init — loaded on first use)
        self.lang_enabled = getattr(cfg.language, 'enabled', False)
        self.lang_extract_every_n = getattr(cfg.language, 'extract_every_n', 5)
        self.lang_weight = getattr(cfg.loss, 'lang_weight', 0.1)
        self.lang_warmup_frames = getattr(cfg.language.autoencoder, 'warmup_frames', 100)
        self._clip_extractor = None
        self._sam_extractor = None
        self._feature_builder = None
        self._autoencoder = None
        self._lang_initialized = False
        # Cache: keyframe_id -> compressed feature map (H, W, latent_dim)
        self._lang_cache: dict[int, torch.Tensor] = {}

        # Dynamic object masking (lazy init)
        self.dynamic_enabled = getattr(cfg.dynamic, 'enabled', False)
        self._dynamic_detector = None
        self._temporal_filter = None
        self._dynamic_initialized = False
        self._dynamic_keyframe_count = 0

        # CoTracker-based (PIPs lineage) class-agnostic dynamic mask.
        # Lecture 27 (Warping and Tracking) explicitly covers PIPs; CoTracker2
        # is the direct descendant from the same Meta lab, available via
        # torch.hub. Mask is UNIONed with YOLO and applied to mapping +
        # Bayesian-belief update + contamination cleanup (NOT tracking — we
        # test its effect on the map first).
        pips_cfg = getattr(cfg.dynamic, 'pips', None)
        self.pips_enabled = bool(
            self.dynamic_enabled and pips_cfg
            and getattr(pips_cfg, 'enabled', False)
        )
        self.pips_window = int(getattr(pips_cfg, 'window_size', 8)) if pips_cfg else 8
        self.pips_grid = int(getattr(pips_cfg, 'grid', 20)) if pips_cfg else 20
        self.pips_thresh_px = float(getattr(pips_cfg, 'thresh_px', 25.0)) if pips_cfg else 25.0
        self.pips_dilation = int(getattr(pips_cfg, 'dilation_kernel', 15)) if pips_cfg else 15
        self._pips_buffer = None
        self._pips_model = None

    def _should_add_keyframe(self, pose: torch.Tensor, frame_id: int) -> bool:
        """Motion-based keyframe selection.

        Adds a keyframe if EITHER:
          * translation since last keyframe >= kf_trans_thresh (meters)
          * rotation    since last keyframe >= kf_rot_thresh   (radians)
          * it has been >= kf_max_interval frames since last keyframe

        Falls back to every-N-frames if no keyframes exist yet.
        """
        if len(self.keyframes) == 0:
            return True

        kf_trans = getattr(self.cfg.slam.keyframe, "kf_trans_thresh", 0.05)   # 5cm
        kf_rot = getattr(self.cfg.slam.keyframe, "kf_rot_thresh", 0.087)      # ~5 deg
        kf_max = getattr(self.cfg.slam.keyframe, "kf_max_interval",
                         self.keyframe_every_n * 3)
        kf_min = getattr(self.cfg.slam.keyframe, "kf_min_interval", 2)

        last_kf = self.keyframes[-1]
        frames_since = frame_id - last_kf["frame_id"]
        if frames_since < kf_min:
            return False
        if frames_since >= kf_max:
            return True

        last_pose = last_kf["pose"]
        # Translation delta
        t_delta = (pose[:3, 3] - last_pose[:3, 3]).norm().item()
        # Rotation delta via trace of relative rotation
        R_rel = pose[:3, :3] @ last_pose[:3, :3].T
        cos_angle = ((R_rel.trace() - 1.0) * 0.5).clamp(-1.0, 1.0)
        r_delta = torch.acos(cos_angle).item()

        return t_delta >= kf_trans or r_delta >= kf_rot

    def _init_dynamic_pipeline(self) -> None:
        """Lazy-initialize the dynamic detection pipeline."""
        if self._dynamic_initialized:
            return

        print("Initializing dynamic object detector...", flush=True)
        from ..dynamic import DynamicDetector, TemporalFilter

        cfg = self.cfg.dynamic
        self._dynamic_detector = DynamicDetector(
            model_name=cfg.yolo_model,
            confidence_thresh=cfg.confidence_thresh,
            dynamic_classes=list(cfg.dynamic_classes),
            device=self.device,
        )
        self._temporal_filter = TemporalFilter(
            window_size=cfg.temporal_window,
            min_detections=cfg.min_detections,
            dilation_kernel=cfg.mask_dilation_kernel,
        )
        self._dynamic_initialized = True
        print("  Dynamic detector ready.", flush=True)

    def process_first_frame(
        self,
        gaussian_map: GaussianMap,
        frame: dict,
    ) -> torch.Tensor:
        """Initialize the map from the first frame.

        Args:
            gaussian_map: Empty Gaussian map to initialize
            frame: dict with 'rgb', 'depth', 'pose'

        Returns:
            pose: (4, 4) first frame pose (ground truth used for initialization)
        """
        pose = frame["pose"].to(self.device)
        rgb = frame["rgb"].to(self.device)
        depth = frame["depth"].to(self.device)

        # Initialize Gaussians from first frame depth
        gaussian_map.initialize_from_depth(
            depth=depth,
            rgb=rgb,
            pose=pose,
            fx=self.intrinsics["fx"],
            fy=self.intrinsics["fy"],
            cx=self.intrinsics["cx"],
            cy=self.intrinsics["cy"],
            downsample=4,
        )

        # Store first keyframe
        self.keyframes.append({
            "frame": {"rgb": rgb, "depth": depth},
            "pose": pose,
            "frame_id": 0,
        })
        self.estimated_poses.append(pose)
        self.frame_count = 1
        self._prev_depth = depth.squeeze(0).clone()

        return pose

    def _init_language_pipeline(self) -> None:
        """Lazy-initialize the language pipeline (CLIP, SAM, autoencoder)."""
        if self._lang_initialized:
            return

        print("Initializing language pipeline...", flush=True)
        from ..language import CLIPExtractor, SAMExtractor, FeatureMapBuilder, LanguageAutoencoder

        cfg = self.cfg.language
        # Load on CPU first; we'll move to GPU only during extraction
        self._clip_extractor = CLIPExtractor(
            model_name=cfg.clip_model,
            pretrained=cfg.clip_pretrained,
            device="cpu",
        )
        self._sam_extractor = SAMExtractor(
            model_type=cfg.sam_model,
            checkpoint_path=getattr(cfg, 'sam_checkpoint', 'checkpoints/sam2.1_hiera_tiny.pt'),
            device="cpu",
        )
        self._feature_builder = FeatureMapBuilder(
            self._clip_extractor, self._sam_extractor, device=self.device,
        )
        self._autoencoder = LanguageAutoencoder(
            input_dim=cfg.autoencoder.input_dim,
            hidden_dim=cfg.autoencoder.hidden_dim,
            latent_dim=cfg.autoencoder.latent_dim,
            lr=cfg.autoencoder.lr,
            device=self.device,
        )
        self._lang_initialized = True
        print("  Language pipeline ready.", flush=True)

    def _extract_language_features(
        self, rgb: torch.Tensor, frame_id: int
    ) -> torch.Tensor | None:
        """Extract and compress language features for a keyframe.

        Args:
            rgb: (3, H, W) tensor [0, 1] on device
            frame_id: frame index

        Returns:
            (H, W, latent_dim) compressed feature map, or None if skipped
        """
        if not self.lang_enabled:
            return None

        self._init_language_pipeline()

        # Convert to numpy uint8 for CLIP/SAM
        rgb_np = (rgb.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

        # Move CLIP + SAM2 to GPU for extraction, then back to CPU
        self._clip_extractor.model.to(self.device)
        self._clip_extractor.device = self.device
        self._sam_extractor.sam2.to(self.device)
        self._feature_builder.device = self.device
        torch.cuda.empty_cache()

        # Build 768-dim feature maps
        # Use configurable scales: single "whole" scale saves ~3x memory
        lang_scales = getattr(self.cfg.language, "scales", ["whole"])
        feat_map = None
        for scale in lang_scales:
            scale_map = self._feature_builder.build_single_scale(rgb_np, scale=scale)
            scale_cpu = scale_map.cpu()
            del scale_map
            torch.cuda.empty_cache()
            if feat_map is None:
                feat_map = scale_cpu
            else:
                feat_map += scale_cpu
        if len(lang_scales) > 1:
            feat_map /= float(len(lang_scales))

        # Offload CLIP + SAM2 back to CPU to free GPU for SLAM
        self._clip_extractor.model.to("cpu")
        self._clip_extractor.device = "cpu"
        self._sam_extractor.sam2.to("cpu")
        self._feature_builder.device = "cpu"
        torch.cuda.empty_cache()

        # Re-normalize after averaging (keep on CPU to save GPU mem)
        feat_map = torch.nn.functional.normalize(feat_map, dim=-1)

        # Flatten and add to autoencoder buffer for training (CPU is fine)
        flat_feats = feat_map.reshape(-1, self._clip_extractor.feat_dim)
        # Subsample every 8th pixel for better spatial diversity
        subsample = flat_feats[::8]
        self._autoencoder.add_features(subsample)

        # Train autoencoder during warmup
        if not self._autoencoder.is_frozen:
            if self._autoencoder.buffer_size >= 512:
                ae_info = self._autoencoder.train_step(batch_size=512, num_steps=50)
                if frame_id % 50 == 0:
                    print(f"  AE train: loss={ae_info['total']:.4f} "
                          f"(l1={ae_info['l1']:.4f}, cos={ae_info['cosine']:.4f}) "
                          f"buf={ae_info['buffer_size']}", flush=True)

                # Freeze when BOTH conditions met:
                # 1. Minimum frames processed (enough scene coverage)
                # 2. Cosine reconstruction is good enough (> 0.85 = 1-0.15)
                cosine_quality = 1.0 - ae_info.get("cosine", 1.0)
                past_warmup = self.frame_count >= self.lang_warmup_frames
                converged = cosine_quality > 0.85 and ae_info["total"] < 0.5

                if past_warmup and converged:
                    self._autoencoder.freeze()
                    print(f"  Autoencoder frozen at frame {self.frame_count} "
                          f"(steps={self._autoencoder._train_steps}, "
                          f"cos_quality={cosine_quality:.3f})", flush=True)
                elif self.frame_count >= self.lang_warmup_frames * 3:
                    # Hard cap: freeze anyway after 3x warmup to avoid training forever
                    self._autoencoder.freeze()
                    print(f"  Autoencoder force-frozen at frame {self.frame_count} "
                          f"(steps={self._autoencoder._train_steps}, "
                          f"cos_quality={cosine_quality:.3f} - did not converge)", flush=True)

        # Compress: (H, W, 768) -> (H, W, latent_dim)
        # Process in chunks to avoid OOM when moving full map to GPU
        with torch.no_grad():
            H, W = feat_map.shape[:2]
            flat = feat_map.reshape(-1, self._clip_extractor.feat_dim)
            # Encode in batches of 8192 to limit GPU memory
            chunk_size = 8192
            chunks = []
            for i in range(0, flat.shape[0], chunk_size):
                chunk = flat[i:i+chunk_size].to(self.device)
                chunks.append(self._autoencoder.encode(chunk))
            compressed = torch.cat(chunks, dim=0)
            compressed_map = compressed.reshape(H, W, -1)

        return compressed_map

    def process_frame(
        self,
        gaussian_map: GaussianMap,
        frame: dict,
        use_gt_pose: bool = False,
    ) -> dict:
        """Process a single frame: track, then optionally map.

        Args:
            gaussian_map: Current Gaussian map
            frame: dict with 'rgb', 'depth', 'pose' (GT pose for evaluation)
            use_gt_pose: if True, skip tracking and use GT pose

        Returns:
            info dict with tracking/mapping metrics
        """
        rgb = frame["rgb"].to(self.device)
        depth = frame["depth"].to(self.device)
        gt_pose = frame["pose"].to(self.device)
        frame_id = frame["frame_id"]

        info = {"frame_id": frame_id}

        # --- DYNAMIC MASKING ---
        dynamic_mask = None  # None = no masking (all pixels used)
        if self.dynamic_enabled:
            self._init_dynamic_pipeline()
            # Raw YOLO output: bool (H, W), True = detected dynamic class
            raw_dynamic = self._dynamic_detector.detect_and_merge(rgb)
            raw_dynamic = raw_dynamic.to(self.device)

            # FIX 6: Depth-verify BEFORE temporal filter.
            # Removes stationary YOLO detections (e.g. sitting person, furniture
            # mislabeled) by requiring the flagged pixel to have actually moved
            # in depth compared to the previous frame.
            depth_verify_thresh = getattr(self.cfg.dynamic, 'depth_verify_thresh', 0.05)
            if self._prev_depth is not None and raw_dynamic.any():
                cur_depth_hw = depth.squeeze(0)      # (H, W)
                prev_depth_hw = self._prev_depth      # (H, W)
                depth_diff = (cur_depth_hw - prev_depth_hw).abs()
                mean_depth = cur_depth_hw.clamp(min=0.1)
                relative_diff = depth_diff / mean_depth
                # Pixels flagged dynamic but with no depth change -> un-flag them
                not_moving = relative_diff < depth_verify_thresh
                raw_dynamic = raw_dynamic & ~not_moving

            # Temporal filter expects CPU bool and returns (H, W) float (1=static, 0=dynamic)
            dynamic_mask = self._temporal_filter.update(raw_dynamic.cpu())
            dynamic_mask = dynamic_mask.to(self.device)

            self._prev_depth = depth.squeeze(0).clone()

            n_dynamic = (dynamic_mask < 0.5).sum().item()
            if n_dynamic > 0:
                info["dynamic_pixels"] = n_dynamic
                info["dynamic_pct"] = n_dynamic / (self.height * self.width) * 100

        # --- TRACKING ---
        t0 = time.time()
        if use_gt_pose:
            est_pose = gt_pose
            info["tracking_loss"] = 0.0
        else:
            # Use constant-velocity model for pose initialization
            use_velocity = getattr(self.cfg.slam.tracking, 'use_velocity_init', True)
            if use_velocity and len(self.estimated_poses) >= 2:
                # Extrapolate: pose_{t} = pose_{t-1} @ (pose_{t-2}^{-1} @ pose_{t-1})
                prev = self.estimated_poses[-1]
                prev2 = self.estimated_poses[-2]
                delta = fast_se3_inverse(prev2) @ prev
                init_pose = prev @ delta
            elif self.estimated_poses:
                init_pose = self.estimated_poses[-1]
            else:
                init_pose = gt_pose

            est_pose, track_info = self.tracker.track(
                gaussian_map=gaussian_map,
                gt_rgb=rgb,
                gt_depth=depth,
                K=self.K,
                width=self.width,
                height=self.height,
                init_pose=init_pose,
                mask=dynamic_mask,
            )
            info["tracking_loss"] = track_info["final_loss"]

            # Motion magnitude clamping: if the estimated pose jumps too far
            # from the previous pose, clamp to a reasonable distance.
            # This prevents catastrophic single-frame drift.
            if len(self.estimated_poses) >= 2:
                prev_t = self.estimated_poses[-1][:3, 3]
                est_t = est_pose[:3, 3]
                cur_motion = (est_t - prev_t).norm().item()

                # Compute average recent motion magnitude
                recent_motions = []
                for j in range(max(0, len(self.estimated_poses) - 10),
                               len(self.estimated_poses) - 1):
                    t1 = self.estimated_poses[j][:3, 3]
                    t2 = self.estimated_poses[j + 1][:3, 3]
                    recent_motions.append((t2 - t1).norm().item())

                if recent_motions:
                    avg_motion = sum(recent_motions) / len(recent_motions)
                    max_allowed = max(avg_motion * 5.0, 0.05)  # 5x avg or 5cm minimum
                    if cur_motion > max_allowed:
                        # Clamp: scale translation to max allowed distance
                        direction = est_t - prev_t
                        clamped_t = prev_t + direction * (max_allowed / cur_motion)
                        est_pose = est_pose.clone()
                        est_pose[:3, 3] = clamped_t

        info["tracking_time"] = time.time() - t0

        # Compute ATE (translation error) against GT
        ate = torch.norm(est_pose[:3, 3] - gt_pose[:3, 3]).item()
        info["ate"] = ate

        self.estimated_poses.append(est_pose)
        self.frame_count += 1

        # --- PIPs / CoTracker class-agnostic mask (post-tracking) ---
        # Runs AFTER tracking so we have the refined pose for the current
        # frame. Mask is UNIONed with the YOLO semantic mask and will be used
        # by mapping, Bayesian belief, and contamination cleanup below.
        if self.pips_enabled:
            # Lazy-init buffer and model on first use (avoid hub download cost
            # until we actually need it).
            if self._pips_buffer is None:
                from ..dynamic import PointTrackBuffer
                self._pips_buffer = PointTrackBuffer(window_size=self.pips_window)
            if self._pips_model is None:
                print("Loading CoTracker2 for class-agnostic dynamic mask...", flush=True)
                self._pips_model = torch.hub.load(
                    "facebookresearch/co-tracker", "cotracker2",
                    trust_repo=True,
                ).to(self.device).eval()
                print("  CoTracker2 ready.", flush=True)

            # Push refined (rgb, depth, tracked-pose) into the buffer.
            self._pips_buffer.push(rgb, depth.squeeze(0), est_pose)

            # When the buffer is full, compute the mask and union it in.
            if self._pips_buffer.is_full():
                from ..dynamic import compute_pips_dynamic_mask
                t_pips = time.time()
                result = compute_pips_dynamic_mask(
                    buffer=self._pips_buffer,
                    K=self.K,
                    model=self._pips_model,
                    grid=self.pips_grid,
                    residual_thresh_px=self.pips_thresh_px,
                    dilation_kernel=self.pips_dilation,
                    semantic_mask=dynamic_mask,
                )
                info["pips_time"] = time.time() - t_pips
                if result is not None:
                    new_mask, pips_info = result
                    dynamic_mask = new_mask
                    info.update(pips_info)
                    # Recompute overall dynamic% from the fused mask.
                    n_dyn = (dynamic_mask < 0.5).sum().item()
                    info["dynamic_pixels"] = n_dyn
                    info["dynamic_pct"] = n_dyn / (self.height * self.width) * 100

        # FIX 5: Update Bayesian dynamic belief for all Gaussians
        if self.dynamic_enabled and dynamic_mask is not None:
            use_bayesian = getattr(self.cfg.dynamic, 'use_bayesian_belief', True)
            if use_bayesian:
                viewmat = fast_se3_inverse(est_pose)
                gaussian_map.update_dynamic_belief(
                    viewmat, self.K, self.width, self.height, dynamic_mask,
                    increase=getattr(self.cfg.dynamic, 'belief_increase', 0.3),
                    decay=getattr(self.cfg.dynamic, 'belief_decay', 0.05),
                )

        # --- KEYFRAME CHECK ---
        # Motion-based keyframe selection: add a keyframe when the camera has
        # moved enough (translation or rotation) since the last keyframe, OR
        # as a hard timeout to guarantee map growth during slow motion.
        is_keyframe = self._should_add_keyframe(est_pose, frame_id)
        info["is_keyframe"] = is_keyframe

        # Advance keyframe counter BEFORE the language trigger check so that
        # extraction cadence (`lang_extract_every_n`) is measured in keyframes,
        # not in raw frames. Required for motion-based KF selection where
        # `frame_id` no longer aligns with predictable multiples.
        if is_keyframe:
            self._keyframe_count += 1

        # --- LANGUAGE FEATURE EXTRACTION ---
        # Extract on every Nth keyframe (keyframe-count based, not frame_id based)
        should_extract_lang = (
            self.lang_enabled
            and is_keyframe
            and (self._keyframe_count % self.lang_extract_every_n == 0)
        )
        if should_extract_lang:
            t_lang = time.time()
            lang_map = self._extract_language_features(rgb, frame_id)
            if lang_map is not None:
                self._lang_cache[frame_id] = lang_map
            info["lang_extract_time"] = time.time() - t_lang

        # --- MAPPING ---
        if is_keyframe:
            t0 = time.time()

            # Store keyframe
            if len(self.keyframes) >= self.max_keyframes:
                oldest = self.keyframes.pop(0)
                # Clean up language cache for evicted keyframes
                evicted_id = oldest.get("frame_id", -1)
                self._lang_cache.pop(evicted_id, None)

            self.keyframes.append({
                "frame": {"rgb": rgb, "depth": depth},
                "pose": est_pose,
                "frame_id": frame_id,
                "dynamic_mask": dynamic_mask,
            })

            # Use recent keyframes for mapping
            window = self.keyframes[-self.keyframe_window_size:]
            map_frames = [kf["frame"] for kf in window]
            map_poses = [kf["pose"] for kf in window]

            # Gather language feature maps for the keyframe window
            lang_maps_for_window = None
            effective_lang_weight = 0.0
            if self.lang_enabled and self._autoencoder is not None:
                lang_maps_for_window = []
                for kf in window:
                    kf_id = kf["frame_id"]
                    if kf_id in self._lang_cache:
                        lang_maps_for_window.append(self._lang_cache[kf_id])
                    else:
                        lang_maps_for_window.append(None)

                # Only apply language loss if we have at least one feature map
                # AND autoencoder is frozen (don't hurt geometry with bad early targets)
                if any(m is not None for m in lang_maps_for_window):
                    if self._autoencoder.is_frozen:
                        # Ramp up over 50 frames after freeze
                        frames_since_freeze = max(0, self.frame_count - self.lang_warmup_frames)
                        ramp = min(1.0, frames_since_freeze / 50.0)
                        effective_lang_weight = self.lang_weight * ramp
                    else:
                        effective_lang_weight = 0.0  # no lang loss during AE warmup
                    # Replace None entries with zeros so mapper can handle them
                    H, W = self.height, self.width
                    latent_dim = self._autoencoder.latent_dim
                    for i, m in enumerate(lang_maps_for_window):
                        if m is None:
                            lang_maps_for_window[i] = torch.zeros(
                                H, W, latent_dim, device=self.device
                            )
                else:
                    lang_maps_for_window = None

            # Gather dynamic masks for the keyframe window
            dynamic_masks_for_window = None
            if self.dynamic_enabled:
                dynamic_masks_for_window = [kf.get("dynamic_mask") for kf in window]

            map_info = self.mapper.map(
                gaussian_map=gaussian_map,
                frames=map_frames,
                poses=map_poses,
                K=self.K,
                width=self.width,
                height=self.height,
                add_new=True,
                lang_feature_maps=lang_maps_for_window,
                lang_weight=effective_lang_weight,
                masks=dynamic_masks_for_window,
            )

            info["mapping_time"] = time.time() - t0
            info["mapping_loss"] = map_info["final_loss"]
            info["gaussians_added"] = map_info["gaussians_added"]
            info["gaussians_pruned"] = map_info["gaussians_pruned"]
            info["total_gaussians"] = map_info["total_gaussians"]
            if "lang_loss" in map_info:
                info["lang_loss"] = map_info["lang_loss"]

            # Local BA writeback: if the mapper refined non-anchor keyframe
            # poses, propagate the correction to (a) the keyframe window so
            # the next mapping call sees the corrected history, and (b) the
            # per-frame estimated_poses so ATE computation reflects the
            # refinement. poses[0] is anchor (unchanged), poses[1:] match
            # the last (K-1) keyframes in self.keyframes.
            if "refined_poses" in map_info:
                refined = map_info["refined_poses"]
                for i, kf in enumerate(window):
                    kf["pose"] = refined[i].detach()
                    kf_id = kf["frame_id"]
                    if 0 <= kf_id < len(self.estimated_poses):
                        self.estimated_poses[kf_id] = refined[i].detach()
                info["pose_refine_trans_max_m"] = map_info.get("pose_refine_trans_max_m", 0.0)
                info["pose_refine_trans_mean_m"] = map_info.get("pose_refine_trans_mean_m", 0.0)

            # Contamination cleanup: periodically remove Gaussians
            # corrupted by dynamic objects
            if self.dynamic_enabled:
                self._dynamic_keyframe_count += 1
                cleanup_every = getattr(self.cfg.dynamic, 'cleanup_every_n', 20)
                if self._dynamic_keyframe_count % cleanup_every == 0:
                    thresh = getattr(self.cfg.dynamic, 'contamination_thresh', 3)
                    n_cleaned = gaussian_map.cleanup_contaminated(thresh)
                    if n_cleaned > 0:
                        info["contamination_cleaned"] = n_cleaned
        else:
            info["mapping_time"] = 0.0
            info["total_gaussians"] = gaussian_map.num_gaussians

        return info

    def compute_ate_rmse(self, gt_poses: list[torch.Tensor] = None) -> float:
        """Compute ATE RMSE over all processed frames.

        Args:
            gt_poses: list of (4,4) ground truth poses. If None, uses stored GT.

        Returns:
            ATE RMSE in meters
        """
        if gt_poses is None or len(self.estimated_poses) == 0:
            return -1.0

        n = min(len(self.estimated_poses), len(gt_poses))
        errors = []
        for i in range(n):
            est_t = self.estimated_poses[i][:3, 3]
            gt_t = gt_poses[i][:3, 3].to(self.device)
            errors.append(torch.norm(est_t - gt_t).item())

        errors = np.array(errors)
        return float(np.sqrt(np.mean(errors ** 2)))

    def query_3d(
        self,
        gaussian_map: GaussianMap,
        text: str,
        top_k: int = 100,
        use_relevancy: bool = True,
    ) -> dict:
        """Query the 3D Gaussian map with a text string.

        Encodes text through CLIP -> autoencoder, then computes relevancy
        score (contrastive vs canonical phrases) against all Gaussians.

        Args:
            gaussian_map: Gaussian map with trained lang_feats
            text: text query (e.g., "chair", "red object")
            top_k: number of top-matching Gaussians to return
            use_relevancy: if True, use LangSplat contrastive relevancy (sharper)

        Returns:
            dict with 'scores' (N,), 'top_k_indices' (K,),
            'top_k_positions' (K, 3), 'top_k_scores' (K,)
        """
        if not self._lang_initialized or self._autoencoder is None:
            raise RuntimeError("Language pipeline not initialized. Run SLAM with language enabled first.")

        # Move CLIP to GPU for text encoding
        self._clip_extractor.model.to(self.device)
        self._clip_extractor.device = self.device

        # Text -> CLIP 768-dim -> autoencoder latent
        text_feat_clip = self._clip_extractor.encode_text(text)  # (768,)
        text_feat_latent = self._autoencoder.encode(text_feat_clip.unsqueeze(0).to(self.device)).squeeze(0)

        with torch.no_grad():
            lang_feats = gaussian_map.lang_feats.data  # (N, latent_dim)
            lang_norm = torch.nn.functional.normalize(lang_feats, dim=-1)
            query_norm = torch.nn.functional.normalize(text_feat_latent.unsqueeze(0), dim=-1)
            sim_query = (lang_norm @ query_norm.T).squeeze(-1)  # (N,)

            if use_relevancy:
                # Contrastive relevancy against canonical phrases
                canonical = ["object", "things", "stuff", "texture"]
                canon_clips = self._clip_extractor.encode_texts(canonical).to(self.device)  # (C, 768)
                canon_latents = self._autoencoder.encode(canon_clips)  # (C, latent_dim)
                canon_norm = torch.nn.functional.normalize(canon_latents, dim=-1)
                sim_canons = lang_norm @ canon_norm.T  # (N, C)
                max_canon = sim_canons.max(dim=-1).values  # (N,)

                # Temperature scaling (must match feature_map.py compute_relevancy)
                temperature = 50.0
                exp_query = torch.exp(temperature * sim_query)
                exp_canon = torch.exp(temperature * max_canon)
                scores = exp_query / (exp_query + exp_canon)  # (N,) in [0, 1]
            else:
                scores = sim_query

            top_k = min(top_k, scores.shape[0])
            top_scores, top_indices = scores.topk(top_k)
            top_positions = gaussian_map.means.data[top_indices]

        # Offload CLIP back to CPU
        self._clip_extractor.model.to("cpu")
        self._clip_extractor.device = "cpu"
        torch.cuda.empty_cache()

        return {
            "scores": scores,
            "top_k_indices": top_indices,
            "top_k_positions": top_positions,
            "top_k_scores": top_scores,
            "query_text": text,
        }
