"""Live text-query demo over a saved DynLang-SLAM map (Tier A showcase).

Opens a Rerun 3D viewer with the reconstructed scene as colored Gaussians.
Type any text query; matching regions light up in place (decoder-route
relevancy, same pathway as query_3d). Each query is a step on the Rerun
timeline, so you can scrub back through previous queries during a demo.

Usage:
    python scripts/demo_live_query.py                       # room0 map, live viewer
    python scripts/demo_live_query.py --checkpoint <path>   # any saved map
    python scripts/demo_live_query.py --save demo.rrd --queries "sofa,lamp,door"
                                                            # headless recording

Checkpoints are produced by scripts/eval_miou_replica.py (and any script
that saves {"map": gmap.state_dict_compact(), "ae": autoencoder.state_dict()}).
"""

import argparse
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
import rerun as rr

from dynlang_slam.core.gaussians import GaussianMap
from dynlang_slam.language import CLIPExtractor, LanguageAutoencoder

CANONICAL = ["object", "things", "stuff", "texture"]
TEMPERATURE = 50.0


def relevancy_scores(decoded, text_feat, canon_feats):
    """LangSplat contrastive relevancy in CLIP space. All inputs normalized."""
    s_q = decoded @ text_feat
    s_c = (decoded @ canon_feats.T).max(dim=-1).values
    e_q = torch.exp(TEMPERATURE * s_q)
    e_c = torch.exp(TEMPERATURE * s_c)
    return e_q / (e_q + e_c)


def turbo(values):
    """Map [0,1] to a blue->red colormap, uint8 (N, 3)."""
    import matplotlib.cm as cm
    return (cm.turbo(values.clamp(0, 1).cpu().numpy())[:, :3] * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(
        PROJECT_ROOT, "results", "sweep", "miou_slam_checkpoint.pt"))
    ap.add_argument("--save", default=None,
                    help="write a .rrd recording instead of opening the viewer")
    ap.add_argument("--queries", default=None,
                    help="comma-separated scripted queries (otherwise interactive)")
    ap.add_argument("--point-radius", type=float, default=0.008)
    ap.add_argument("--up", default="y-down",
                    choices=["y-down", "y-up", "z-up", "z-down"],
                    help="world up-axis convention so the room sits level "
                         "on the viewer grid (BONN/TUM maps: y-down; try "
                         "z-up if the scene still looks tilted)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, weights_only=False)

    gmap = GaussianMap(sh_degree=0, lang_feat_dim=16, init_opacity=0.5,
                       device=device)
    gmap.load_state_dict_compact(ckpt["map"])
    ae = LanguageAutoencoder(input_dim=768, hidden_dim=256, latent_dim=16,
                             device=device)
    ae.load_state_dict(ckpt["ae"])
    ae.freeze()
    print(f"Map: {gmap.num_gaussians} Gaussians")

    print("Loading CLIP (text encoder)...")
    clip = CLIPExtractor(model_name="ViT-L-14", pretrained="openai",
                         device=device)

    with torch.no_grad():
        positions = gmap.means.data.cpu().numpy()
        base_colors = (gmap.colors.data.clamp(0, 1).cpu().numpy() * 255
                       ).astype(np.uint8)
        norm_ok = gmap.lang_feats.data.norm(dim=-1) > 0.1
        decoded = F.normalize(ae.decode(gmap.lang_feats.data), dim=-1)
        canon = F.normalize(clip.encode_texts(CANONICAL).to(device), dim=-1)

    rr.init("DynLang-SLAM_live_query")
    if args.save:
        rr.save(args.save)
    else:
        rr.spawn()

    view_coords = {
        "y-down": rr.ViewCoordinates.RIGHT_HAND_Y_DOWN,
        "y-up": rr.ViewCoordinates.RIGHT_HAND_Y_UP,
        "z-up": rr.ViewCoordinates.RIGHT_HAND_Z_UP,
        "z-down": rr.ViewCoordinates.RIGHT_HAND_Z_DOWN,
    }[args.up]
    rr.log("/", view_coords, static=True)
    rr.log("map/scene", rr.Points3D(positions, colors=base_colors,
                                    radii=args.point_radius), static=True)
    print(f"\nScene logged ({int(norm_ok.sum())} of {gmap.num_gaussians} "
          f"Gaussians queryable). Type a query, or 'quit'.")

    def run_query(text, step):
        prompt = f"a photo of a {text} in a room"
        with torch.no_grad():
            tf = F.normalize(
                clip.encode_text(prompt).to(device).unsqueeze(0), dim=-1
            ).squeeze(0)
            rel = relevancy_scores(decoded, tf, canon)
            rel = torch.where(norm_ok, rel, torch.zeros_like(rel))

        rr.set_time("query", sequence=step)
        # Per-query contrast normalization for display (raw values go to the
        # log): min-max over queryable Gaussians so the match visually pops.
        v = rel[norm_ok]
        rel_vis = ((rel - v.min()) / (v.max() - v.min() + 1e-8)).clamp(0, 1)
        rel_vis = torch.where(norm_ok, rel_vis, torch.zeros_like(rel_vis))
        colors = turbo(rel_vis)
        radii = (args.point_radius * (0.25 + 1.75 * rel_vis.cpu().numpy())
                 ).astype(np.float32)
        rr.log("map/query_heat", rr.Points3D(positions, colors=colors,
                                             radii=radii))
        top = rel.topk(min(100, rel.shape[0]))
        top_pos = gmap.means.data[top.indices]
        center = top_pos.mean(dim=0).cpu().numpy()
        # Localization coherence, scale-free: fraction of the top-100 matches
        # within 0.75 m of the single best match. A real object clusters; a
        # query for something NOT in the map (e.g. a masked-out person)
        # scatters across the room. Robust to scene size and duplicate
        # object instances (unlike a raw spread cutoff).
        best = top_pos[0]
        cluster_frac = float(
            ((top_pos - best).norm(dim=-1) < 0.75).float().mean().item())
        spread = float(top_pos.std(dim=0).norm().item())
        rr.log("map/top_center", rr.Points3D(center.reshape(1, 3),
                                             colors=[[255, 255, 255]],
                                             radii=[0.06]))
        rr.log("query_log", rr.TextLog(
            f"'{text}'  cluster={cluster_frac:.0%} spread={spread:.2f}m  "
            f"max={rel.max():.3f}  "
            f"center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})"))
        print(f"  '{text}': cluster={cluster_frac:.0%} spread={spread:.2f}m "
              f"max={rel.max():.3f}")

    if args.queries:
        for i, q in enumerate([q.strip() for q in args.queries.split(",")]):
            run_query(q, i)
        print("Scripted queries done.")
        return

    step = 0
    while True:
        try:
            q = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q.lower() in ("quit", "exit"):
            break
        run_query(q, step)
        step += 1


if __name__ == "__main__":
    main()
