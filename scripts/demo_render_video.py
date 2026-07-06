"""Render showcase videos/GIFs of a saved map with text-query heatmaps.

Produces a turntable orbit of the actual Gaussian rendering (not a point
cloud): first segment in true color, then one segment per query with the
map recolored by decoder-route relevancy (turbo) and the query burned in
as a label. Outputs an .mp4 and a README-friendly .gif.

Usage:
    python scripts/demo_render_video.py \
        --checkpoint results/sweep/demo_bonn_checkpoint.pt \
        --intrinsics bonn --queries "monitor,chair,person" \
        --out results/media/demo_bonn
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F

from dynlang_slam.core.gaussians import GaussianMap
from dynlang_slam.core.renderer import GaussianRenderer
from dynlang_slam.language import CLIPExtractor, LanguageAutoencoder

CANONICAL = ["object", "things", "stuff", "texture"]
TEMPERATURE = 50.0


def lookat_c2w(eye, center, world_down):
    """OpenCV-convention camera-to-world (x right, y down, z forward)."""
    z = center - eye
    z = z / z.norm()
    y = world_down - torch.dot(world_down, z) * z
    y = y / y.norm()
    x = torch.linalg.cross(y, z)
    c2w = torch.eye(4, device=eye.device)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = x, y, z, eye
    return c2w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--intrinsics", choices=["bonn", "replica"], default="bonn")
    ap.add_argument("--queries", default="monitor,chair,person")
    ap.add_argument("--out", required=True, help="output basename (no ext)")
    ap.add_argument("--frames", type=int, default=48, help="frames per segment")
    ap.add_argument("--downscale", type=int, default=2)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--elev", type=float, default=0.35,
                    help="camera height above center, in scene radii")
    ap.add_argument("--dist", type=float, default=1.6,
                    help="orbit distance, in scene radii")
    ap.add_argument("--gif-width", type=int, default=440)
    ap.add_argument("--gif-stride", type=int, default=3)
    ap.add_argument("--mode", choices=["orbit", "lookaround"], default="orbit",
                    help="orbit: circle the scene from outside (partial "
                         "scans). lookaround: stand inside and pan 360 "
                         "(room-interior scans, where an outside orbit "
                         "would only show the backs of walls)")
    args = ap.parse_args()

    device = "cuda"
    ckpt = torch.load(args.checkpoint, weights_only=False)
    gmap = GaussianMap(sh_degree=0, lang_feat_dim=16, init_opacity=0.5,
                       device=device)
    gmap.load_state_dict_compact(ckpt["map"])
    ae = LanguageAutoencoder(input_dim=768, hidden_dim=256, latent_dim=16,
                             device=device)
    ae.load_state_dict(ckpt["ae"])
    ae.freeze()
    clip = CLIPExtractor(model_name="ViT-L-14", pretrained="openai",
                         device=device)
    renderer = GaussianRenderer(near=0.01, far=100.0)

    if args.intrinsics == "bonn":
        from dynlang_slam.data.tum import get_bonn_intrinsics
        intr = get_bonn_intrinsics()
    else:
        from dynlang_slam.data.replica import get_replica_intrinsics
        intr = get_replica_intrinsics()
    K = intr["K"].to(device)
    W, H = intr["width"], intr["height"]

    with torch.no_grad():
        pos = gmap.means.data
        center = pos.median(dim=0).values
        radius = (pos - center).norm(dim=-1).quantile(0.9).item()
        norm_ok = gmap.lang_feats.data.norm(dim=-1) > 0.1
        decoded = F.normalize(ae.decode(gmap.lang_feats.data), dim=-1)
        canon = F.normalize(clip.encode_texts(CANONICAL).to(device), dim=-1)
        true_colors = gmap.colors.data.clone()

    def heat_colors(text):
        import matplotlib.cm as cm
        with torch.no_grad():
            tf = F.normalize(clip.encode_text(
                f"a photo of a {text} in a room").to(device).unsqueeze(0),
                dim=-1).squeeze(0)
            s_q = decoded @ tf
            s_c = (decoded @ canon.T).max(dim=-1).values
            rel = torch.exp(TEMPERATURE * s_q) / (
                torch.exp(TEMPERATURE * s_q) + torch.exp(TEMPERATURE * s_c))
            rel = torch.where(norm_ok, rel, torch.zeros_like(rel))
            v = rel[norm_ok]
            vis = ((rel - v.min()) / (v.max() - v.min() + 1e-8)).clamp(0, 1)
            rgb = torch.from_numpy(
                cm.turbo(vis.cpu().numpy())[:, :3]).float().to(device)
        return rgb

    world_down = torch.tensor([0.0, 1.0, 0.0], device=device)
    segments = [("", true_colors)]
    for q in [q.strip() for q in args.queries.split(",") if q.strip()]:
        segments.append((q, heat_colors(q)))

    frames = []
    total = len(segments) * args.frames
    print(f"rendering {total} frames ({len(segments)} segments)...")
    with torch.no_grad():
        for seg_i, (label, colors) in enumerate(segments):
            gmap.colors.data = colors
            for f_i in range(args.frames):
                t = (seg_i * args.frames + f_i) / max(total - 1, 1)
                az = 2 * np.pi * t          # one full turn across the video
                if args.mode == "orbit":
                    eye = center + torch.tensor([
                        args.dist * radius * np.cos(az),
                        -args.elev * radius,
                        args.dist * radius * np.sin(az),
                    ], device=device, dtype=torch.float32)
                    target = center
                else:  # lookaround: stand near the center, pan outward
                    eye = center + torch.tensor(
                        [0.0, -args.elev * radius * 0.3, 0.0],
                        device=device, dtype=torch.float32)
                    target = center + torch.tensor([
                        radius * np.cos(az),
                        0.0,
                        radius * np.sin(az),
                    ], device=device, dtype=torch.float32)
                c2w = lookat_c2w(eye, target, world_down)
                rendered = renderer(gmap, torch.inverse(c2w), K, W, H,
                                    render_lang=False,
                                    downscale=args.downscale)
                img = (rendered["rgb"].clamp(0, 1).cpu().numpy() * 255
                       ).astype(np.uint8)
                if label:
                    cv2.putText(img, f'query: "{label}"', (16, 34),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
                    cv2.putText(img, f'query: "{label}"', (16, 34),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (255, 255, 255), 2)
                else:
                    cv2.putText(img, "reconstructed map", (16, 34),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
                    cv2.putText(img, "reconstructed map", (16, 34),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                (255, 255, 255), 2)
                frames.append(img)
    gmap.colors.data = true_colors

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    h, w = frames[0].shape[:2]

    mp4_path = args.out + ".mp4"
    vw = cv2.VideoWriter(mp4_path, cv2.VideoWriter_fourcc(*"mp4v"),
                         args.fps, (w, h))
    for f in frames:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()

    gif_path = args.out + ".gif"
    gif_w = args.gif_width
    gif_frames = [cv2.resize(f, (gif_w, int(h * gif_w / w)))
                  for f in frames[::args.gif_stride]]
    imageio.mimsave(gif_path, gif_frames,
                    fps=max(args.fps // args.gif_stride, 6), loop=0)

    print(f"wrote {mp4_path} ({os.path.getsize(mp4_path)/1e6:.1f} MB)")
    print(f"wrote {gif_path} ({os.path.getsize(gif_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
