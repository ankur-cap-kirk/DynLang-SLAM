"""Multi-seed statistical sweep (plan E1, week 1).

3 scenes x 3 modes x 8 seeds = 72 runs, each in its OWN subprocess so
VRAM leaks cannot accumulate across runs (the single-process d16 protocol
OOM'd after ~7 runs on the 8GB GPU). Results append to a JSONL, so the
sweep is resumable: rerunning the driver skips completed combos.

Usage:
    python sweep_multiseed.py --drive          # run all missing combos
    python sweep_multiseed.py --run h1 full 42 # one combo (driver calls this)
    python sweep_multiseed.py --summarize      # mean +/- std table

Rationale: identical code produced ATE differing by 4-6 cm across runs on
BONN (nondeterministic rasterizer atomics), so single-run comparisons are
noise. n=8 gives every paper table a real error bar.
"""

import argparse
import json
import os
import subprocess
import sys
import time

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPTS, ".."))
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, PROJECT_ROOT)

SWEEP_DIR = os.path.join(PROJECT_ROOT, "results", "sweep")
RESULTS_JSONL = os.path.join(SWEEP_DIR, "sweep_results.jsonl")

SCENE_TAGS = ["h1", "h2", "h3"]
MODES = ["static", "dynamic", "full"]
SEEDS = [42, 123, 2024, 7, 77, 555, 3407, 31337]
PER_RUN_TIMEOUT_S = 3600


def load_done() -> set:
    done = set()
    if os.path.exists(RESULTS_JSONL):
        with open(RESULTS_JSONL) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" not in r:
                    done.add((r["scene"], r["mode"], r["seed"]))
    return done


def append_result(result: dict) -> None:
    os.makedirs(SWEEP_DIR, exist_ok=True)
    with open(RESULTS_JSONL, "a") as f:
        f.write(json.dumps(result) + "\n")


def run_single(tag: str, mode: str, seed: int) -> None:
    """Execute one (scene, mode, seed) combo in this process."""
    import rerun_d16_all as R

    scene = next(s for s in R.SCENES if s["tag"] == tag)
    result = R.run_one(scene, mode=mode, seed=seed, save_traj=False)
    append_result(result)


def drive() -> None:
    done = load_done()
    combos = [(t, m, s) for t in SCENE_TAGS for m in MODES for s in SEEDS
              if (t, m, s) not in done]
    total = len(SCENE_TAGS) * len(MODES) * len(SEEDS)
    print(f"sweep: {len(done)}/{total} already done, {len(combos)} to run",
          flush=True)

    for i, (tag, mode, seed) in enumerate(combos, 1):
        label = f"{tag}/{mode}/seed{seed}"
        t0 = time.time()
        print(f"[{i}/{len(combos)}] START {label}", flush=True)
        proc = subprocess.run(
            [sys.executable, os.path.abspath(__file__),
             "--run", tag, mode, str(seed)],
            cwd=os.path.dirname(PROJECT_ROOT),  # parent dir (yolo weights)
            capture_output=True, text=True, timeout=PER_RUN_TIMEOUT_S,
        )
        dt = time.time() - t0
        if proc.returncode == 0:
            # child appended its own result line; read it back for the log
            print(f"[{i}/{len(combos)}] DONE  {label}  ({dt:.0f}s)", flush=True)
        else:
            tail = (proc.stderr or "")[-500:]
            print(f"[{i}/{len(combos)}] FAILED {label} ({dt:.0f}s): {tail}",
                  flush=True)
            append_result({"scene": tag, "mode": mode, "seed": seed,
                           "error": tail or f"exit {proc.returncode}"})
    print("SWEEP COMPLETE", flush=True)
    summarize()


def summarize() -> None:
    import numpy as np
    by = {}
    if not os.path.exists(RESULTS_JSONL):
        print("no results yet")
        return
    with open(RESULTS_JSONL) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in r:
                continue
            by.setdefault((r["scene"], r["mode"]), []).append(r["ate_rmse_cm"])
    print(f"\n{'scene':>6} {'mode':>8} {'n':>3} {'ATE mean+/-std (cm)':>22} "
          f"{'min':>6} {'max':>6}")
    for tag in SCENE_TAGS:
        for mode in MODES:
            vals = by.get((tag, mode))
            if not vals:
                continue
            v = np.asarray(vals)
            print(f"{tag:>6} {mode:>8} {len(v):>3} "
                  f"{v.mean():>12.2f} +/- {v.std():>5.2f} "
                  f"{v.min():>6.2f} {v.max():>6.2f}")


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--drive", action="store_true")
    g.add_argument("--run", nargs=3, metavar=("SCENE", "MODE", "SEED"))
    g.add_argument("--summarize", action="store_true")
    args = ap.parse_args()

    if args.drive:
        drive()
    elif args.run:
        run_single(args.run[0], args.run[1], int(args.run[2]))
    else:
        summarize()


if __name__ == "__main__":
    main()
