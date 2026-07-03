"""R1 (trajectory plot) + R2 (per-frame aligned ATE vs time) figures for paper.

Reads per-scene trajectory NPZs produced by `scripts/export_trajectories.py`
and produces two publication-quality figures under `results/figures/`.
Pure CPU / matplotlib -- fast iteration on plot style without re-running SLAM.

R1  trajectory.pdf/.png
    1x3 grid (h1, h2, h3). For each scene, top-down 2D projection
    (configurable axes; default XZ) showing:
        - GT trajectory (solid black, thick)
        - Ours aligned (dashed blue, thinner)
        - start marker (filled circle), end marker (X)
    Axes in meters, equal aspect, per-scene titles with aligned ATE-RMSE.

R2  ate_vs_frame.pdf/.png
    1x3 grid (h1, h2, h3). For each scene, line plot of
    per-frame aligned translation error (cm) vs frame index, with
    the RMSE plotted as a horizontal dashed line.

Usage:
    python scripts/figures_r1_r2.py                    # default XZ projection
    python scripts/figures_r1_r2.py --axes xy          # override projection
    python scripts/figures_r1_r2.py --scenes h1,h3     # subset

CVPR-style output conventions:
    - Vector PDF primary (embeds cleanly in LaTeX)
    - 300 DPI PNG fallback
    - 8pt base font, 9pt titles (matches CVPR \\small rendering)
    - serif font family (Times-equivalent) to match CVPR template
"""
import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR = os.path.join(PROJECT_ROOT, "results", "figures")

# ---- Scene metadata -----------------------------------------------------
# Keyed by scene tag (h1/h2/h3). Display name is what ends up in the
# figure panel title; NPZ glob is how we find the export from disk.
SCENE_META = {
    "h1": dict(
        display="BONN person\\_tracking",
        npz_glob="traj_h1_*_seed*.npz",
        color="#1f77b4",  # muted blue
    ),
    "h2": dict(
        display="BONN balloon",
        npz_glob="traj_h2_*_seed*.npz",
        color="#d62728",  # muted red
    ),
    "h3": dict(
        display="Replica room0",
        npz_glob="traj_h3_*_seed*.npz",
        color="#2ca02c",  # muted green
    ),
}


def _umeyama_se3(src: np.ndarray, dst: np.ndarray):
    """Horn/Umeyama SE(3) alignment (rotation + translation, unit scale).

    Identical to SLAMPipeline._umeyama_se3 but pulled out to avoid a CUDA
    import on plot-only machines. If you change one, change both.
    """
    assert src.shape == dst.shape and src.shape[1] == 3
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    H = src_c.T @ dst_c / src.shape[0]
    U, _, Vt = np.linalg.svd(H)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = Vt.T @ S @ U.T
    t = mu_dst - R @ mu_src
    return R, t


def _load_trajectories(scene_tag: str):
    """Find + load ALL NPZs for a scene tag (one per seed)."""
    import glob
    pattern = os.path.join(FIG_DIR, SCENE_META[scene_tag]["npz_glob"])
    matches = sorted(glob.glob(pattern))
    if not matches:
        sys.exit(
            f"[figures_r1_r2] no NPZ matching {pattern}. "
            f"Run scripts/export_trajectories.py first."
        )
    return [np.load(m, allow_pickle=True) for m in matches], matches


def _align_and_errors(est_c2w: np.ndarray, gt_c2w: np.ndarray):
    """Umeyama-align est translation to gt translation, return aligned +
    per-frame translation errors (meters)."""
    est_t = est_c2w[:, :3, 3]
    gt_t = gt_c2w[:, :3, 3]
    R, t = _umeyama_se3(est_t, gt_t)
    est_aligned = (R @ est_t.T).T + t
    errors = np.linalg.norm(est_aligned - gt_t, axis=1)
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    return est_aligned, gt_t, errors, rmse


def _set_cvpr_style():
    """Apply CVPR-friendly matplotlib defaults."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,   # TrueType (editor-safe, no Type 3 bitmap)
        "ps.fonttype": 42,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.4,
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.2,
    })


# ---- R1: trajectory plot -------------------------------------------------

_AXIS_IDX = {"x": 0, "y": 1, "z": 2}


def plot_r1_trajectories(scenes, axes_str: str):
    """Draw GT + aligned est trajectory per scene, side by side."""
    ax1 = _AXIS_IDX[axes_str[0]]
    ax2 = _AXIS_IDX[axes_str[1]]
    axis_labels = {0: "X (m)", 1: "Y (m)", 2: "Z (m)"}

    fig, axs = plt.subplots(
        1, len(scenes), figsize=(2.2 * len(scenes) + 0.3, 2.4),
        constrained_layout=True,
    )
    if len(scenes) == 1:
        axs = [axs]

    for ax, scene_tag in zip(axs, scenes):
        runs, paths = _load_trajectories(scene_tag)
        color = SCENE_META[scene_tag]["color"]
        display = SCENE_META[scene_tag]["display"]

        # GT (thick black) -- identical across seeds, pull from first run
        gt_c2w = runs[0]["gt_c2w"]
        gt_t = gt_c2w[:, :3, 3]
        ax.plot(gt_t[:, ax1], gt_t[:, ax2],
                color="black", lw=1.5, label="Ground Truth", zorder=2)

        # Per-seed aligned trajectories, plus compute RMSEs
        rmses = []
        for i, run in enumerate(runs):
            est_aligned, _, _, rmse = _align_and_errors(run["est_c2w"], run["gt_c2w"])
            rmses.append(rmse)
            label = f"Ours (n={len(runs)} seeds)" if i == 0 else None
            ax.plot(est_aligned[:, ax1], est_aligned[:, ax2],
                    color=color, lw=0.8, linestyle="--", alpha=0.55,
                    label=label, zorder=3)

        # Start + end markers on GT
        ax.scatter([gt_t[0, ax1]], [gt_t[0, ax2]],
                   marker="o", c="black", s=22, zorder=4,
                   edgecolors="white", linewidths=0.6)
        ax.scatter([gt_t[-1, ax1]], [gt_t[-1, ax2]],
                   marker="X", c="black", s=26, zorder=4,
                   edgecolors="white", linewidths=0.6)

        mean_rmse_cm = np.mean(rmses) * 100
        std_rmse_cm = np.std(rmses) * 100
        title_stat = (f"ATE = {mean_rmse_cm:.2f} ± {std_rmse_cm:.2f} cm"
                      if len(runs) > 1 else f"ATE = {mean_rmse_cm:.2f} cm")
        ax.set_title(f"{display}\n{title_stat}")
        ax.set_xlabel(axis_labels[ax1])
        ax.set_ylabel(axis_labels[ax2])
        ax.set_aspect("equal", adjustable="datalim")

    # One legend for the whole row (first panel)
    axs[0].legend(loc="best", frameon=True, framealpha=0.9, fancybox=False)

    for ext in ("pdf", "png"):
        out = os.path.join(FIG_DIR, f"R1_trajectory.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  wrote {out}")
    plt.close(fig)


# ---- R2: per-frame ATE vs frame index -----------------------------------

def plot_r2_ate_vs_frame(scenes):
    fig, axs = plt.subplots(
        1, len(scenes), figsize=(2.2 * len(scenes) + 0.3, 2.0),
        constrained_layout=True, sharey=False,
    )
    if len(scenes) == 1:
        axs = [axs]

    for ax, scene_tag in zip(axs, scenes):
        runs, _ = _load_trajectories(scene_tag)
        color = SCENE_META[scene_tag]["color"]
        display = SCENE_META[scene_tag]["display"]

        # Per-seed error curves (stacked for mean/stdev)
        all_err = []
        rmses = []
        for run in runs:
            _, _, errors, rmse = _align_and_errors(run["est_c2w"], run["gt_c2w"])
            all_err.append(errors)
            rmses.append(rmse)
        # Truncate to shortest (in case a run had a frame-count mismatch)
        min_len = min(len(e) for e in all_err)
        err_stack = np.stack([e[:min_len] for e in all_err], axis=0) * 100  # (S, N)
        frames = np.arange(min_len)

        err_mean = err_stack.mean(axis=0)
        err_std = err_stack.std(axis=0)
        mean_rmse_cm = np.mean(rmses) * 100
        std_rmse_cm = np.std(rmses) * 100

        if len(runs) > 1:
            ax.fill_between(frames, err_mean - err_std, err_mean + err_std,
                            color=color, alpha=0.22, linewidth=0,
                            label=f"±1σ across {len(runs)} seeds")
        ax.plot(frames, err_mean, color=color, lw=1.1,
                label="mean per-frame" if len(runs) > 1 else "per-frame")
        rmse_label = (f"RMSE = {mean_rmse_cm:.2f} ± {std_rmse_cm:.2f} cm"
                      if len(runs) > 1 else f"RMSE = {mean_rmse_cm:.2f} cm")
        ax.axhline(mean_rmse_cm, color="black", lw=0.8, linestyle=":",
                   label=rmse_label)
        ax.set_title(display)
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Aligned trans. error (cm)")
        ax.set_ylim(bottom=0)
        ax.legend(loc="best", frameon=True, framealpha=0.9, fancybox=False)

    for ext in ("pdf", "png"):
        out = os.path.join(FIG_DIR, f"R2_ate_vs_frame.{ext}")
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  wrote {out}")
    plt.close(fig)


# ---- Driver --------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--axes", default="xz", choices=["xy", "xz", "yz"],
                   help="Top-down projection plane for R1 (default xz).")
    p.add_argument("--scenes", default="h1,h2,h3",
                   help="Comma-separated subset (default h1,h2,h3).")
    args = p.parse_args()

    scenes = [s.strip() for s in args.scenes.split(",") if s.strip() in SCENE_META]
    if not scenes:
        sys.exit("No valid scenes specified.")

    _set_cvpr_style()
    os.makedirs(FIG_DIR, exist_ok=True)

    print("[R1] trajectory plot")
    plot_r1_trajectories(scenes, args.axes)
    print("[R2] per-frame ATE vs frame")
    plot_r2_ate_vs_frame(scenes)
    print("\n[done]")


if __name__ == "__main__":
    main()
