"""Test corrected loss schedule on room0, room1, room2."""
import sys
import os
sys.path.insert(0, ".")
os.environ["PYTHONUNBUFFERED"] = "1"
from evaluate import run_scene

N_FRAMES = 200
scenes = ["room0", "room1", "room2"]
baselines = {"room0": 1.43, "room1": 8.59, "room2": 3.73}

print(f"\n=== Corrected Loss Schedule Test ({N_FRAMES} frames) ===\n", flush=True)

results = {}
for scene in scenes:
    print(f"Testing: {scene}", flush=True)
    m = run_scene(scene, N_FRAMES, cfg_overrides={})
    ate = m["ate_rmse_cm"]
    gs = m["total_gaussians"]
    psnr = m["psnr"]
    fps = m["fps"]
    results[scene] = ate
    bl = baselines[scene]
    delta = ate - bl
    tag = "IMPROVED" if delta < 0 else "REGRESSED"
    print(f"  -> {scene}: ATE={ate:.2f}cm (baseline={bl:.2f}cm, {delta:+.2f}cm {tag}) | Gaussians={gs} | PSNR={psnr:.1f}\n", flush=True)

print("\n=== Summary ===")
for s in scenes:
    print(f"  {s}: {results[s]:.2f}cm (baseline: {baselines[s]:.2f}cm)")
