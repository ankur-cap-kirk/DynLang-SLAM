"""Coarse-to-fine tracking ablation on room0 and room1."""
import sys
import os
sys.path.insert(0, ".")
os.environ["PYTHONUNBUFFERED"] = "1"
from evaluate import run_scene

N_FRAMES = 200

tests = [
    # Current default: coarse_to_fine=True, coarse_downscale=4, coarse_ratio=0.6
    ("room0/c2f=on",  "room0", {}),
    ("room0/c2f=off", "room0", {"slam.tracking.coarse_to_fine": False}),
    ("room1/c2f=on",  "room1", {}),
    ("room1/c2f=off", "room1", {"slam.tracking.coarse_to_fine": False}),
]

print(f"\n=== Coarse-to-Fine Tracking Ablation ({N_FRAMES} frames) ===\n", flush=True)
results = []
for label, scene, overrides in tests:
    print(f"\nTesting: {label}", flush=True)
    m = run_scene(scene, N_FRAMES, cfg_overrides=overrides)
    ate = m["ate_rmse_cm"]
    gs = m["total_gaussians"]
    psnr = m["psnr"]
    print(f"  -> ATE: {ate:.2f}cm | Gaussians: {gs} | PSNR: {psnr:.1f}", flush=True)
    results.append((label, ate, gs, psnr))

print("\n\n=== Summary ===")
print(f"{'Config':<25} {'ATE(cm)':>8} {'Gaussians':>10} {'PSNR':>6}")
print(f"{'-'*25} {'-'*8} {'-'*10} {'-'*6}")
for label, ate, gs, psnr in results:
    print(f"{label:<25} {ate:>8.2f} {gs:>10} {psnr:>6.1f}")
print(f"\nPrevious baselines:")
print(f"  room0: 1.43cm (iso=0.05, depth_w=1.2)")
print(f"  room1: ~8-10cm")
