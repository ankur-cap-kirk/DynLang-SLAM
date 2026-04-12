"""Verify that DynLang-SLAM setup is complete and all dependencies work."""

import sys


def check(name: str, test_fn) -> bool:
    try:
        result = test_fn()
        print(f"  [OK] {name}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        return False


def main():
    print("=" * 60)
    print("DynLang-SLAM Setup Verification")
    print("=" * 60)
    results = []

    # 1. PyTorch + CUDA
    print("\n1. PyTorch & CUDA")
    results.append(check("PyTorch", lambda: f"v{__import__('torch').__version__}"))
    results.append(check("CUDA available", lambda: (
        f"Yes - {__import__('torch').cuda.get_device_name(0)}"
        if __import__('torch').cuda.is_available()
        else "No (CPU only - will be slow!)"
    )))
    results.append(check("CUDA version", lambda: __import__('torch').version.cuda or "N/A"))

    # 2. Gaussian Splatting Rasterizer (gsplat)
    print("\n2. 3DGS Rasterizer")
    results.append(check("gsplat", lambda: f"v{__import__('gsplat').__version__}"))

    # 3. CLIP
    print("\n3. Language Models")
    results.append(check("open_clip", lambda: f"v{__import__('open_clip').__version__}"))

    # 4. SAM
    results.append(check("segment_anything", lambda: (
        __import__('segment_anything'), "installed"
    )[1]))

    # 5. YOLOv8
    print("\n4. Dynamic Detection")
    results.append(check("ultralytics (YOLOv8)", lambda: f"v{__import__('ultralytics').__version__}"))

    # 6. Evaluation tools
    print("\n5. Evaluation Tools")
    results.append(check("evo (trajectory eval)", lambda: f"v{__import__('evo').__version__}"))
    results.append(check("lpips", lambda: (__import__('lpips'), "installed")[1]))

    # 7. Other deps
    print("\n6. Other Dependencies")
    results.append(check("omegaconf", lambda: f"v{__import__('omegaconf').__version__}"))
    results.append(check("opencv", lambda: f"v{__import__('cv2').__version__}"))
    # open3d: optional, not available on Python 3.14
    try:
        import open3d
        results.append(check("open3d", lambda: f"v{open3d.__version__}"))
    except ImportError:
        print("  [SKIP] open3d: not available (Python 3.14 incompatible, optional)")
    results.append(check("einops", lambda: f"v{__import__('einops').__version__}"))

    # 8. GPU Memory
    print("\n7. GPU Memory")
    try:
        import torch
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            print(f"  [INFO] Total VRAM: {total:.1f} GB")
            if total >= 12:
                print(f"  [OK] Sufficient VRAM for full pipeline")
            elif total >= 8:
                print(f"  [WARN] Tight on VRAM - may need to reduce batch size or use ViT-B CLIP")
            else:
                print(f"  [WARN] Low VRAM - consider using smaller models")
    except Exception:
        pass

    # 8. Dataset
    print("\n8. Dataset")
    from pathlib import Path
    replica_path = Path("./data/Replica")
    if replica_path.exists():
        scenes = [d.name for d in replica_path.iterdir() if d.is_dir()]
        print(f"  [OK] Replica dataset found: {', '.join(scenes)}")
    else:
        print(f"  [PENDING] Replica not downloaded. Run: python scripts/download_replica.py")

    # Summary
    n_pass = sum(results)
    n_total = len(results)
    print(f"\n{'=' * 60}")
    print(f"Results: {n_pass}/{n_total} checks passed")
    if n_pass == n_total:
        print("All dependencies verified! Ready to run DynLang-SLAM.")
    else:
        print("Some dependencies are missing. Install them before proceeding.")
    print(f"{'=' * 60}")

    return 0 if n_pass == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
