"""Preprocess the H4 in-the-wild Pixel 8 capture into TUM-format clips.

The raw input is a single ~6-minute, 4K portrait, 60 fps MP4. The DynLang-SLAM
pipeline expects three things per sequence directory:

    sequence/
        rgb/               # PNG frames at the working resolution
        depth/             # 16-bit PNG depth (mm * depth_scale)
        rgb.txt            # "<timestamp> rgb/<file>" one per line
        depth.txt          # "<timestamp> depth/<file>" one per line
        intrinsics.txt     # "fx fy cx cy" on the first non-comment line

Because the source video has no sensor depth and no GT trajectory we:
    (1) resample to a manageable 30 fps,
    (2) rotate to portrait, downsample to ``target_height x target_width``,
    (3) generate metric depth with DepthAnything-V2 (outdoor variant),
    (4) split the video into three short clips (default 13 s each) chosen
        either automatically by motion-energy peaks or via ``--clips``.

Typical use (one-shot, auto picks clips):

    python scripts/prepare_h4.py \
        --video data/H4_raw/PXL_20260501_222309116.mp4 \
        --output data/H4

To preview motion energy and pick clips manually:

    python scripts/prepare_h4.py --video <path> --output data/H4 --probe-only
    # inspect data/H4/_probe/motion_energy.png, then:
    python scripts/prepare_h4.py --video <path> --output data/H4 \
        --clips "0:35-0:48,2:10-2:23,4:30-4:43"
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

# matplotlib import is lazy so the script runs without it for everything
# except probe-plot generation.
# torch/transformers are imported inside _load_depth_model so probe-only
# runs work on machines without a GPU or HuggingFace cache.


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Preprocess the H4 Pixel video into TUM-format clips."
    )
    p.add_argument("--video", type=str, required=True,
                   help="Path to the raw .mp4 file.")
    p.add_argument("--output", type=str, default="data/H4",
                   help="Output base directory; clip_a/b/c are written under it.")
    p.add_argument("--clips", type=str, default=None,
                   help='Comma-separated time windows "MM:SS-MM:SS,...". '
                        'If omitted, clips are picked automatically from motion peaks.')
    p.add_argument("--clip-duration", type=float, default=13.0,
                   help="Seconds per clip when auto-picking (default 13).")
    p.add_argument("--target-fps", type=float, default=30.0,
                   help="Output frame rate after temporal subsampling (default 30).")
    p.add_argument("--target-height", type=int, default=854,
                   help="Output frame height after resize (default 854 for 16:9 portrait).")
    p.add_argument("--target-width", type=int, default=480,
                   help="Output frame width after resize (default 480).")
    p.add_argument("--depth-model", type=str,
                   default="depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf",
                   help="HuggingFace model id for metric depth.")
    p.add_argument("--depth-scale", type=float, default=5000.0,
                   help="Depth scale for 16-bit PNG storage (default 5000, TUM convention).")
    p.add_argument("--probe-only", action="store_true",
                   help="Compute and plot motion energy, then exit.")
    p.add_argument("--skip-depth", action="store_true",
                   help="Extract RGB only; useful for fast iteration on clip choice.")
    p.add_argument("--device", type=str, default="cuda",
                   help='"cuda" or "cpu" for depth model.')
    p.add_argument("--depth-batch-size", type=int, default=4,
                   help="Mini-batch for depth inference (lower if you OOM).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Time-window helpers
# ---------------------------------------------------------------------------

@dataclass
class ClipWindow:
    name: str
    start_s: float
    end_s: float

    @property
    def duration(self) -> float:
        return self.end_s - self.start_s


def _parse_time(s: str) -> float:
    """Accept ``MM:SS``, ``M:SS`` or a bare number of seconds."""
    s = s.strip()
    if ":" in s:
        m, sec = s.split(":")
        return int(m) * 60 + float(sec)
    return float(s)


def parse_clip_arg(arg: str) -> list[ClipWindow]:
    """Parse ``"0:35-0:48,2:10-2:23,..."`` into ClipWindow objects."""
    windows: list[ClipWindow] = []
    for i, chunk in enumerate(arg.split(",")):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" not in chunk:
            raise ValueError(f"Bad clip spec '{chunk}': expected START-END.")
        start_s, end_s = chunk.split("-", 1)
        windows.append(ClipWindow(
            name=f"clip_{chr(ord('a') + i)}",
            start_s=_parse_time(start_s),
            end_s=_parse_time(end_s),
        ))
    if not windows:
        raise ValueError("--clips parsed to zero windows.")
    return windows


# ---------------------------------------------------------------------------
# Motion-energy probe
# ---------------------------------------------------------------------------

def probe_motion_energy(
    video_path: Path,
    sample_every_n_frames: int = 30,
    probe_long_edge: int = 240,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (sampled_times_s, motion_energy, fps)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[probe] {video_path.name}: {n_total} frames at {fps:.2f} fps "
          f"({n_total / max(fps, 1):.1f} s)")

    times: list[float] = []
    energies: list[float] = []
    prev_gray: np.ndarray | None = None
    idx = 0
    t0 = time.time()
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % sample_every_n_frames == 0:
            _, frame = cap.retrieve()
            if frame is None:
                idx += 1
                continue
            h, w = frame.shape[:2]
            scale = probe_long_edge / max(h, w)
            small = cv2.resize(frame, (max(1, int(w * scale)),
                                       max(1, int(h * scale))))
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
            if prev_gray is not None:
                e = float(np.mean(np.abs(gray - prev_gray)))
                times.append(idx / fps)
                energies.append(e)
            prev_gray = gray
        idx += 1
    cap.release()
    print(f"[probe] sampled {len(energies)} frames in {time.time() - t0:.1f} s")
    return np.asarray(times, dtype=np.float32), np.asarray(energies, dtype=np.float32), fps


def auto_pick_clips(
    times: np.ndarray,
    energies: np.ndarray,
    duration_s: float,
    n_clips: int = 3,
    min_gap_s: float = 45.0,
) -> list[ClipWindow]:
    """Greedily pick non-overlapping windows centred at the highest-energy samples."""
    smoothed = np.convolve(
        energies, np.ones(5, dtype=np.float32) / 5.0, mode="same"
    )
    order = np.argsort(-smoothed)  # high to low
    chosen: list[ClipWindow] = []
    used_centres: list[float] = []
    for i in order:
        centre = float(times[i])
        if any(abs(centre - c) < min_gap_s for c in used_centres):
            continue
        start = max(0.0, centre - duration_s / 2)
        end = start + duration_s
        if end > float(times[-1]):
            continue
        used_centres.append(centre)
        chosen.append(ClipWindow(
            name=f"clip_{chr(ord('a') + len(chosen))}",
            start_s=start,
            end_s=end,
        ))
        if len(chosen) == n_clips:
            break
    chosen.sort(key=lambda c: c.start_s)
    for i, c in enumerate(chosen):
        c.name = f"clip_{chr(ord('a') + i)}"
    return chosen


def save_probe_plot(
    times: np.ndarray,
    energies: np.ndarray,
    clips: list[ClipWindow],
    out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(times, energies, color="#1f77b4", linewidth=0.8, label="motion energy")
    for c in clips:
        ax.axvspan(c.start_s, c.end_s, alpha=0.25, color="#d62728")
        ax.text((c.start_s + c.end_s) / 2, ax.get_ylim()[1] * 0.95,
                c.name, ha="center", va="top", fontsize=9, color="#7a1414")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("|frame diff| (mean)")
    ax.set_title(f"H4 motion energy with {len(clips)} clip windows")
    ax.legend(loc="upper right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"[probe] plot -> {out_path}")


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------

def _orient_portrait(frame: np.ndarray) -> np.ndarray:
    """Rotate landscape frames to portrait by 90 deg CW. Already-portrait passes through."""
    h, w = frame.shape[:2]
    if w > h:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


def extract_clip_frames(
    video_path: Path,
    clip: ClipWindow,
    target_fps: float,
    target_h: int,
    target_w: int,
    out_rgb_dir: Path,
) -> list[tuple[float, str]]:
    """Extract RGB frames for one clip. Returns list of (timestamp_s, filename)."""
    out_rgb_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_MSEC, clip.start_s * 1000.0)

    keep_step = max(1.0, src_fps / target_fps)
    next_keep = 0.0
    in_clip_idx = 0
    written: list[tuple[float, str]] = []

    while True:
        cur_t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if cur_t > clip.end_s:
            break
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if in_clip_idx >= next_keep:
            next_keep += keep_step
            frame = _orient_portrait(frame)
            frame = cv2.resize(frame, (target_w, target_h),
                               interpolation=cv2.INTER_AREA)
            t_s = cur_t  # absolute time in the source video; OK for SLAM
            fname = f"{t_s:010.6f}.png"
            cv2.imwrite(str(out_rgb_dir / fname), frame,
                        [cv2.IMWRITE_PNG_COMPRESSION, 3])
            written.append((t_s, fname))
        in_clip_idx += 1

    cap.release()
    if not written:
        raise RuntimeError(
            f"Extracted 0 frames for {clip.name} ({clip.start_s:.1f}-{clip.end_s:.1f}s). "
            "Check that the time window is inside the video duration."
        )
    print(f"[extract] {clip.name}: wrote {len(written)} RGB frames "
          f"(target_fps={target_fps}, src_fps={src_fps:.2f})")
    return written


# ---------------------------------------------------------------------------
# DepthAnything-V2 metric depth
# ---------------------------------------------------------------------------

def _load_depth_model(model_id: str, device: str):
    try:
        import torch  # noqa: F401
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
    except ImportError as e:
        raise SystemExit(
            "Depth generation needs torch and transformers; "
            "install them or rerun with --skip-depth.\n"
            f"Original error: {e}"
        )
    print(f"[depth] loading {model_id} on {device}...")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model.eval()
    if device == "cuda":
        import torch
        if not torch.cuda.is_available():
            print("[depth] CUDA unavailable, falling back to CPU.")
            device = "cpu"
    model = model.to(device)
    return processor, model, device


def generate_depth_for_clip(
    rgb_dir: Path,
    rgb_files: list[tuple[float, str]],
    out_depth_dir: Path,
    depth_scale: float,
    processor,
    model,
    device: str,
    target_h: int,
    target_w: int,
    batch_size: int,
) -> list[tuple[float, str]]:
    """Run DepthAnything-V2 on RGB frames; write 16-bit PNG depth."""
    import torch

    out_depth_dir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[float, str]] = []
    n = len(rgb_files)
    t0 = time.time()

    for batch_start in range(0, n, batch_size):
        batch = rgb_files[batch_start: batch_start + batch_size]
        imgs = []
        for _, fname in batch:
            img = cv2.imread(str(rgb_dir / fname))
            if img is None:
                raise RuntimeError(f"Failed to read {rgb_dir / fname}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)

        inputs = processor(images=imgs, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            pred = outputs.predicted_depth  # (B, h', w'), metres for Metric variants

        for (t_s, fname), depth_map in zip(batch, pred):
            depth_np = depth_map.cpu().float().numpy()
            depth_np = cv2.resize(depth_np, (target_w, target_h),
                                  interpolation=cv2.INTER_LINEAR)
            depth_np = np.clip(depth_np, 0.0, 80.0)  # 80 m hard cap (outdoor reasonable)
            depth_u16 = np.round(depth_np * depth_scale).astype(np.uint16)
            out_name = fname.replace(".png", ".png")  # mirror RGB filename
            cv2.imwrite(str(out_depth_dir / out_name), depth_u16,
                        [cv2.IMWRITE_PNG_COMPRESSION, 3])
            written.append((t_s, out_name))

        if (batch_start // batch_size) % 10 == 0:
            done = batch_start + len(batch)
            print(f"[depth]  {done}/{n} ({done / max(1, time.time() - t0):.1f} fps)")

    print(f"[depth] {n} frames in {time.time() - t0:.1f} s")
    return written


# ---------------------------------------------------------------------------
# Per-clip writer
# ---------------------------------------------------------------------------

def _write_index(path: Path, entries: list[tuple[float, str]], subdir: str) -> None:
    with open(path, "w") as f:
        f.write(f"# DynLang-SLAM H4 - {subdir} index\n")
        f.write("# timestamp filename\n")
        for t, name in entries:
            f.write(f"{t:.6f} {subdir}/{name}\n")


def _write_intrinsics(path: Path, height: int, width: int) -> None:
    """Write Pixel-8 portrait intrinsics for the given working resolution.

    Mirrors get_pixel8_portrait_intrinsics in dynlang_slam/data/tum.py.
    """
    scale = 2160.0 / float(width)
    fx = 2156.0 / scale
    fy = 2156.0 / scale
    cx = width / 2.0
    cy = height / 2.0
    with open(path, "w") as f:
        f.write("# Pixel 8 main camera, portrait, downsampled\n")
        f.write("# fx fy cx cy\n")
        f.write(f"{fx:.4f} {fy:.4f} {cx:.4f} {cy:.4f}\n")


def write_clip_dataset(
    clip_dir: Path,
    rgb_files: list[tuple[float, str]],
    depth_files: list[tuple[float, str]] | None,
    target_h: int,
    target_w: int,
) -> None:
    _write_index(clip_dir / "rgb.txt", rgb_files, "rgb")
    if depth_files is not None:
        _write_index(clip_dir / "depth.txt", depth_files, "depth")
    _write_intrinsics(clip_dir / "intrinsics.txt", target_h, target_w)
    # Marker so SLAM runner knows GT is unavailable.
    (clip_dir / "NO_GT.txt").write_text(
        "This sequence was preprocessed by scripts/prepare_h4.py from raw\n"
        "phone video. There is no ground-truth trajectory; ATE numbers\n"
        "computed against the identity placeholder pose are meaningless.\n"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    out_base = Path(args.output).expanduser().resolve()
    if not video_path.exists():
        print(f"[error] video not found: {video_path}", file=sys.stderr)
        return 2
    out_base.mkdir(parents=True, exist_ok=True)

    times, energies, fps = probe_motion_energy(video_path)

    if args.clips is not None:
        clips = parse_clip_arg(args.clips)
    else:
        clips = auto_pick_clips(times, energies, args.clip_duration)
        if not clips:
            print("[error] auto-pick failed (video too short?). "
                  "Pass --clips manually.", file=sys.stderr)
            return 2
        print("[probe] auto-picked clips:")
        for c in clips:
            print(f"   {c.name}: {c.start_s:.1f} s - {c.end_s:.1f} s")

    save_probe_plot(times, energies, clips, out_base / "_probe" / "motion_energy.png")

    if args.probe_only:
        print("[probe] --probe-only set; exiting before extraction.")
        return 0

    processor = model = None
    device = args.device
    if not args.skip_depth:
        processor, model, device = _load_depth_model(args.depth_model, args.device)

    for clip in clips:
        clip_dir = out_base / clip.name
        clip_dir.mkdir(parents=True, exist_ok=True)
        rgb_files = extract_clip_frames(
            video_path, clip,
            target_fps=args.target_fps,
            target_h=args.target_height,
            target_w=args.target_width,
            out_rgb_dir=clip_dir / "rgb",
        )

        depth_files: list[tuple[float, str]] | None = None
        if not args.skip_depth:
            depth_files = generate_depth_for_clip(
                rgb_dir=clip_dir / "rgb",
                rgb_files=rgb_files,
                out_depth_dir=clip_dir / "depth",
                depth_scale=args.depth_scale,
                processor=processor,
                model=model,
                device=device,
                target_h=args.target_height,
                target_w=args.target_width,
                batch_size=args.depth_batch_size,
            )

        write_clip_dataset(
            clip_dir,
            rgb_files=rgb_files,
            depth_files=depth_files,
            target_h=args.target_height,
            target_w=args.target_width,
        )
        print(f"[done] {clip.name} -> {clip_dir}")

    print("\nNext steps:")
    print(f"  1. Inspect {out_base / '_probe' / 'motion_energy.png'} and a few PNGs in "
          f"{out_base / clips[0].name / 'rgb'}.")
    print("  2. Run the smoke test:")
    print(f"       python scripts/h4_smoke_test.py "
          f"--clip {out_base / clips[0].name}")
    print("  3. If smoke passes, run full SLAM via configs/h4.yaml.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
