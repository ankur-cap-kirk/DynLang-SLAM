"""Tracking-and-mapping fly-by visualisation, WildGS-SLAM style.

Renders the trained Gaussian map from an external "fly-by" camera and
overlays the estimated camera frusta + trajectory polyline. The result
is a single figure showing both the reconstructed scene and the camera
path through it, like the reference figure the supervisor asked for.

Inputs:
    results/figures/filmstrip_h1/map_ckpt_h1.pt   (made by filmstrip.py)

Outputs:
    results/figures/figure_tracking_mapping.{pdf,png}

Run:
    python scripts/figure_tracking_mapping.py
"""
import os
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from gsplat import rasterization

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIG_DIR  = os.path.join(PROJECT_ROOT, "results", "figures")

# Scene selection (env var DYNLANG_SCENE = h1 | h2 | h3 | h4). All four
# checkpoints land under results/figures/filmstrip_<scene>/map_ckpt_<scene>.pt
# once filmstrip.py has been run on that scene.
SCENE = os.environ.get("DYNLANG_SCENE", "h1").lower()
SCENE_TITLE = {
    "h1": "BONN person\\_tracking",
    "h2": "BONN balloon",
    "h3": "Replica room0",
    "h4": "H4 in-the-wild (Pixel 8)",
}.get(SCENE, SCENE)
FILM_DIR = os.path.join(FIG_DIR, f"filmstrip_{SCENE}")
CKPT     = os.path.join(FILM_DIR, f"map_ckpt_{SCENE}.pt")

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ── Helpers ───────────────────────────────────────────────────────────────
def look_at(eye, target, up=(0.0, 1.0, 0.0)):
    """Build a c2w (camera-to-world) matrix for an OpenCV-convention
    camera (x right, y DOWN, z forward). `up` is the WORLD up direction
    (against gravity).

    Construction:
        z = forward = (target - eye) normalized
        x = right   = up × z         normalized
        y = down    = x × z          (NOT z × x — that gives world-up)

    The y = x × z order is what makes the rendered image right-side-up:
    in OpenCV image space +Y is down, so c2w[:,1] must equal world-down
    = (x × z), not world-up = (z × x).
    """
    eye = np.asarray(eye, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    up = np.asarray(up, dtype=np.float32)
    z = target - eye
    z /= (np.linalg.norm(z) + 1e-9)
    x = np.cross(up, z)
    x /= (np.linalg.norm(x) + 1e-9)
    y = np.cross(x, z)                            # camera-down
    R = np.stack([x, y, z], axis=1)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R
    c2w[:3, 3]  = eye
    return c2w


def fast_inv(T):
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float32)
    Ti[:3, :3] = R.T
    Ti[:3, 3]  = -R.T @ t
    return Ti


def project_points(viewmat_w2c, K, pts_world):
    """Project Nx3 world points to Nx2 pixel coords via the given
    world-to-camera matrix and intrinsics K (3x3)."""
    n = pts_world.shape[0]
    homo = np.concatenate([pts_world, np.ones((n, 1))], axis=1)  # (n,4)
    cam = (viewmat_w2c @ homo.T).T[:, :3]                        # (n,3)
    valid = cam[:, 2] > 1e-3
    cam[~valid] = np.nan
    pix = (K @ cam.T).T
    pix[:, 0] /= pix[:, 2]
    pix[:, 1] /= pix[:, 2]
    return pix[:, :2], valid


def frustum_corners_local(K, width, height, depth=0.18):
    """Five frustum corners in camera-local frame (apex at origin +
    four image-plane corners projected forward by `depth`)."""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    # image plane corners as pixels → rays (z=1 plane) → scaled by depth
    px = np.array([0, width, width, 0])
    py = np.array([0, 0, height, height])
    rx = (px - cx) / fx
    ry = (py - cy) / fy
    plane = np.stack([rx, ry, np.ones_like(rx)], axis=1) * depth
    apex  = np.zeros((1, 3), dtype=np.float32)
    return np.concatenate([apex, plane.astype(np.float32)], axis=0)


def transform_points(c2w, pts_local):
    n = pts_local.shape[0]
    homo = np.concatenate([pts_local, np.ones((n, 1))], axis=1)
    return (c2w @ homo.T).T[:, :3]


# ── Load checkpoint ───────────────────────────────────────────────────────
if not os.path.exists(CKPT):
    raise FileNotFoundError(
        f"Checkpoint not found at {CKPT}.\n"
        "Run scripts/filmstrip.py first to generate it.")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[load] {CKPT}  device={device}")
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
snap = ckpt["snapshot"]
poses_c2w = ckpt["estimated_poses"].numpy().astype(np.float32)  # (N,4,4)
K_orig = ckpt["K"].numpy().astype(np.float32)
W_orig, H_orig = int(ckpt["width"]), int(ckpt["height"])
N_poses = poses_c2w.shape[0]
print(f"[load] {N_poses} poses · {snap['n']} gaussians · "
      f"render {W_orig}x{H_orig}")


# ── Choose external "fly-by" viewpoint ────────────────────────────────────
# Centroid + spread of estimated camera positions tells us where the
# scene is and how big it is.
cam_pos = poses_c2w[:, :3, 3]                       # (N,3)
center  = cam_pos.mean(axis=0)
spread  = (cam_pos.max(axis=0) - cam_pos.min(axis=0))
spread_norm = float(np.linalg.norm(spread))

# Derive the world-up direction directly from the SLAM camera poses:
# in OpenCV convention each camera's local +Y points DOWN, so world-up
# is the average of (-c2w[:, :3, 1]) across all poses. This is robust
# regardless of whether BONN's ground-truth uses Y-up or Z-up.
cam_ups = -poses_c2w[:, :3, 1]                # (N, 3)
WORLD_UP = cam_ups.mean(axis=0)
WORLD_UP /= (np.linalg.norm(WORLD_UP) + 1e-9)
# Mean forward direction the SLAM cameras were looking at — that's
# roughly where the reconstructed scene Gaussians live.
cam_fwds = poses_c2w[:, :3, 2]                # (N, 3)
mean_fwd_raw = cam_fwds.mean(axis=0)
mean_fwd_raw /= (np.linalg.norm(mean_fwd_raw) + 1e-9)
# Project the mean forward onto the horizontal plane (perpendicular to
# world-up) so the external view is level instead of tilted up/down. This
# makes the rendered scene "face the viewer" rather than slanting away.
mean_fwd = mean_fwd_raw - (mean_fwd_raw @ WORLD_UP) * WORLD_UP
mean_fwd /= (np.linalg.norm(mean_fwd) + 1e-9)
# Side axis perpendicular to (horizontal) forward and up
side_dir = np.cross(WORLD_UP, mean_fwd)
side_dir /= (np.linalg.norm(side_dir) + 1e-9)
print(f"[view] world_up={WORLD_UP.round(2)}  "
      f"mean_fwd={mean_fwd.round(2)}  "
      f"side_dir={side_dir.round(2)}")

# Position the external camera like WildGS-SLAM's reference: nearly at
# the SLAM cameras' own height, slightly back and to the side, only a
# small elevation. Aim toward the scene the cameras are looking at, so
# the wall / objects fill the frame and the trajectory sits across the
# foreground.
back = 0.80 * spread_norm + 0.55
side = 0.50 * spread_norm + 0.35
elev = 0.10 * spread_norm + 0.06    # camera-level, only a touch above

eye = (center
       - mean_fwd * back        # step back from the trajectory
       + side_dir * side         # off to one side
       + WORLD_UP * elev)        # tiny elevation
# Aim well into the scene so the wall (where the SLAM cameras were
# looking) becomes the dominant content of the rendered image.
target = center + mean_fwd * (1.50 * spread_norm + 1.00)
ext_c2w = look_at(eye.tolist(), target.tolist(),
                  up=tuple(WORLD_UP))
ext_w2c = fast_inv(ext_c2w)
print(f"[view] center={center.round(2)} spread={spread.round(2)} "
      f"eye={eye.round(2)}")


# ── Render Gaussian map from external viewpoint ───────────────────────────
# Use a higher-res render for the figure (1.5x the SLAM resolution).
RENDER_SCALE = 1.4
W = int(W_orig * RENDER_SCALE)
H = int(H_orig * RENDER_SCALE)
K = K_orig.copy()
K[0, :] *= RENDER_SCALE
K[1, :] *= RENDER_SCALE

means     = snap["means"].to(device)
scales    = torch.exp(snap["scales"].to(device))
quats_raw = snap["quats"].to(device)
quats     = quats_raw / (quats_raw.norm(dim=-1, keepdim=True) + 1e-9)
opac      = torch.sigmoid(snap["opacities"].to(device)).squeeze(-1)
colors    = snap["colors"].to(device)
if colors.dim() == 3:
    colors = colors.squeeze(1)
colors = colors.clamp(0.0, 1.0)

viewmat_t = torch.from_numpy(ext_w2c).to(device).unsqueeze(0)
K_t       = torch.from_numpy(K).to(device).unsqueeze(0)

with torch.no_grad():
    out, alpha, _ = rasterization(
        means, quats, scales, opac, colors,
        viewmat_t, K_t, W, H,
        render_mode="RGB",
    )
# Composite onto a soft off-white background so empty regions (no
# Gaussian coverage) read as background rather than black.
rgb = out[0].clamp(0, 1).cpu().numpy()
a = alpha[0].clamp(0, 1).cpu().numpy()
bg = 0.97
img = rgb * a + bg * (1 - a)
print(f"[render] {W}x{H} fly-by view rendered")


# ── Project trajectory + frusta into the fly-by image ─────────────────────
# Trajectory polyline
traj_pix, traj_valid = project_points(ext_w2c, K, cam_pos)

# Frusta — pick the N most spatially-spread keyframes via greedy
# farthest-point sampling on camera positions, instead of uniform time
# sampling. This gives the best visual separation in scenes with
# limited camera motion.
def _farthest_point_sample(points, n):
    """Greedy farthest-point sampling: start from index 0, then
    iteratively pick the point with the largest min-distance to the
    already-selected set."""
    n = min(n, points.shape[0])
    sel = [0, points.shape[0] - 1]   # always anchor first + last
    while len(sel) < n:
        d = np.min(
            np.linalg.norm(points[:, None, :] - points[sel][None, :, :],
                           axis=2),
            axis=1,
        )
        nxt = int(np.argmax(d))
        if nxt in sel:
            break
        sel.append(nxt)
    return np.array(sorted(set(sel)))

n_frusta = 5
sel_idx = _farthest_point_sample(cam_pos, n_frusta)
print(f"[frusta] keyframe indices selected (FPS): {sel_idx.tolist()}")
local_corners = frustum_corners_local(K_orig, W_orig, H_orig,
                                      depth=0.07 * spread_norm + 0.045)

frustum_lines = []   # list of (Nx2) pixel polylines
labels_to_draw = []  # (text, pix_xy)
for i, idx in enumerate(sel_idx):
    c2w = poses_c2w[idx]
    world_corners = transform_points(c2w, local_corners)
    pix, valid = project_points(ext_w2c, K, world_corners)
    if not valid.all():
        continue
    apex = pix[0]
    p1, p2, p3, p4 = pix[1], pix[2], pix[3], pix[4]
    # Apex → 4 corners
    for p in (p1, p2, p3, p4):
        frustum_lines.append(np.stack([apex, p], axis=0))
    # Image-plane rectangle
    frustum_lines.append(np.stack([p1, p2, p3, p4, p1], axis=0))
    # Label only first / mid / last, with vertical offsets so labels
    # don't stack on top of each other when frusta are clustered
    if i == 0:
        labels_to_draw.append((f"Cam. {idx+1}", (apex[0] - 60, apex[1] - 14)))
    elif i == n_frusta // 2:
        labels_to_draw.append((f"Cam. {idx+1}", (apex[0] + 12, apex[1] + 22)))
    elif i == n_frusta - 1:
        labels_to_draw.append((f"Cam. {idx+1}", (apex[0] + 12, apex[1] - 14)))


# ── Compose figure ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6.5))
ax.imshow(img, interpolation="bilinear")
ax.set_xlim(0, W)
ax.set_ylim(H, 0)

# Trajectory polyline (red)
xs = traj_pix[:, 0]
ys = traj_pix[:, 1]
mask = ~np.isnan(xs)
ax.plot(xs[mask], ys[mask], color="#D93030", lw=2.0,
        zorder=4, solid_capstyle="round")

# Frusta (blue wireframes)
for poly in frustum_lines:
    ax.plot(poly[:, 0], poly[:, 1],
            color="#1F4FA8", lw=1.0, zorder=5)

# Camera labels
for txt, (x, y) in labels_to_draw:
    ax.text(x, y, txt, color="#1F4FA8", fontsize=10,
            fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round,pad=0.18",
                      facecolor="#FFFFFFEE",
                      edgecolor="#1F4FA8", lw=0.6))

# Legend
legend_handles = [
    Line2D([0], [0], color="#D93030", lw=2, label="Estimated trajectory"),
    Line2D([0], [0], color="#1F4FA8", lw=1.2, label="Camera frustum (keyframe)"),
]
ax.legend(handles=legend_handles, loc="lower right",
          framealpha=0.9, fontsize=9)

ax.set_title("Tracking & Mapping  —  reconstructed static map + estimated trajectory",
             fontsize=11, fontweight="bold", pad=14)
ax.text(8, H - 12,
        f"DynLang-SLAM on {SCENE_TITLE} · "
        f"{snap['n']} Gaussians · {N_poses} frames",
        color="#222", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.25",
                  facecolor="#FFFFFFD0",
                  edgecolor="#999", lw=0.4))
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

# ── Save (per-scene) ─────────────────────────────────────────────────────
out_pdf = os.path.join(FIG_DIR, f"figure_tracking_mapping_{SCENE}.pdf")
out_png = os.path.join(FIG_DIR, f"figure_tracking_mapping_{SCENE}.png")
plt.savefig(out_pdf, bbox_inches="tight", pad_inches=0.05)
plt.savefig(out_png, bbox_inches="tight", pad_inches=0.05, dpi=240)
plt.close()
print(f"Saved {out_pdf}")
print(f"Saved {out_png}")
