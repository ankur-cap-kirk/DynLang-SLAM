"""Iso reg weight ablation on room0."""
import sys
import os
sys.path.insert(0, ".")
os.environ["PYTHONUNBUFFERED"] = "1"
from evaluate import run_scene

N_FRAMES = 200

tests = [
    ("room0/iso=0.05",  "room0", {}),
    ("room0/iso=0.01",  "room0", {"slam.mapping.iso_reg_weight": 0.01}),
    ("room0/iso=0",     "room0", {"slam.mapping.iso_reg_weight": 0.0}),
]

print(f"\n=== Iso Reg Ablation ({N_FRAMES} frames) ===\n", flush=True)
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
print(f"\nOriginal baseline (no iso, no densify, no vel): 0.72cm")
