"""Download the Replica dataset (NICE-SLAM preprocessed version).

This downloads the same version used by SplaTAM, SGS-SLAM, and SemGauss-SLAM
for direct comparison. Hosted by ETH Zurich CVG group.

Usage:
    python scripts/download_replica.py --output_dir ./data/Replica
"""

import argparse
import os
import subprocess
import sys
import zipfile
from pathlib import Path


REPLICA_URL = "https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip"
EXPECTED_SCENES = [
    "room0", "room1", "room2",
    "office0", "office1", "office2", "office3", "office4",
]


def download_with_progress(url: str, output_path: str) -> None:
    """Download a file with progress bar using curl or wget."""
    if subprocess.run(["curl", "--version"], capture_output=True).returncode == 0:
        subprocess.run(
            ["curl", "-L", "-o", output_path, "--progress-bar", url],
            check=True,
        )
    elif subprocess.run(["wget", "--version"], capture_output=True).returncode == 0:
        subprocess.run(
            ["wget", "-O", output_path, "--show-progress", url],
            check=True,
        )
    else:
        # Fallback to Python urllib
        import urllib.request

        def _progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            pct = min(100.0, downloaded * 100.0 / total_size) if total_size > 0 else 0
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Downloading: {mb:.1f}/{total_mb:.1f} MB ({pct:.1f}%)")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, output_path, _progress)
        print()


def verify_dataset(data_dir: Path) -> bool:
    """Verify that the dataset was extracted correctly."""
    ok = True
    for scene in EXPECTED_SCENES:
        scene_dir = data_dir / scene
        if not scene_dir.exists():
            print(f"  [MISSING] {scene}")
            ok = False
            continue

        # Check for key files
        results_dir = scene_dir / "results"
        if not results_dir.exists():
            print(f"  [INCOMPLETE] {scene} - missing results/")
            ok = False
            continue

        # Count frames
        frame_files = list(results_dir.glob("frame*.jpg"))
        depth_files = list(results_dir.glob("depth*.png"))
        print(f"  [OK] {scene}: {len(frame_files)} RGB frames, {len(depth_files)} depth maps")

    return ok


def main():
    parser = argparse.ArgumentParser(description="Download Replica dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/Replica",
        help="Output directory for the dataset",
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip download, only verify existing data",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    zip_path = output_dir.parent / "Replica.zip"

    if not args.skip_download:
        # Create output directory
        output_dir.parent.mkdir(parents=True, exist_ok=True)

        if zip_path.exists():
            print(f"Zip file already exists at {zip_path}")
            print("Delete it and re-run to re-download, or use --skip_download")
        else:
            print(f"Downloading Replica dataset from ETH Zurich...")
            print(f"URL: {REPLICA_URL}")
            print(f"This is ~5GB and may take a few minutes.\n")
            download_with_progress(REPLICA_URL, str(zip_path))
            print(f"\nDownload complete: {zip_path}")

        # Extract
        if not output_dir.exists() or not any(output_dir.iterdir()):
            print(f"\nExtracting to {output_dir}...")
            with zipfile.ZipFile(str(zip_path), "r") as zf:
                zf.extractall(str(output_dir.parent))
            print("Extraction complete.")
        else:
            print(f"\nDataset directory already exists at {output_dir}")

    # Verify
    print(f"\nVerifying dataset at {output_dir}...")
    if verify_dataset(output_dir):
        print("\nDataset verified successfully!")
        print(f"\nTo use in DynLang-SLAM, set in config:")
        print(f"  dataset.path: \"{output_dir}\"")
        print(f"  dataset.scene: \"room0\"  # or any of: {', '.join(EXPECTED_SCENES)}")
    else:
        print("\nDataset verification failed. Some scenes may be missing.")
        return 1

    # Optionally clean up zip
    if zip_path.exists():
        size_gb = zip_path.stat().st_size / (1024**3)
        print(f"\nTip: You can delete {zip_path} ({size_gb:.1f}GB) to save disk space.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
