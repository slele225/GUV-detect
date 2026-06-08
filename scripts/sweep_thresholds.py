"""Run detect_real.py at several detection thresholds in a single invocation.

Each threshold runs the full real-image transfer check and writes to its own
folder following the detect_real.py convention -- results/thresh_<value>/ -- so
you can compare thresholds side by side without any run overwriting another.

    # default sweep [0.10, 0.15, 0.20, 0.25, 0.30] on models/best.pt + data/
    uv run python scripts/sweep_thresholds.py

    # custom thresholds / inputs
    uv run python scripts/sweep_thresholds.py --thresholds 0.1 0.2 0.3
    uv run python scripts/sweep_thresholds.py --checkpoint models/best.pt --images data \
        --thresholds 0.15 0.25 --nms-dist 8 --device cpu

Each threshold is dispatched to scripts/detect_real.py as a subprocess so the
inference + output convention is byte-for-byte identical to a single run.
"""

import argparse
import subprocess
import sys
from pathlib import Path

DEFAULT_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30]


def main():
    repo_root = Path(__file__).resolve().parents[1]
    detect_real = repo_root / "scripts" / "detect_real.py"

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--thresholds", type=float, nargs="+", default=DEFAULT_THRESHOLDS,
                        help=f"list of heatmap thresholds (default: {DEFAULT_THRESHOLDS})")
    parser.add_argument("--checkpoint", default=str(repo_root / "models" / "best.pt"),
                        help="trained model (default: models/best.pt)")
    parser.add_argument("--images", default=str(repo_root / "data"),
                        help="folder of real images (default: data/)")
    parser.add_argument("--nms-dist", type=float, default=6.0, help="NMS center distance (px)")
    parser.add_argument("--channel", type=int, default=1, help="lipid channel if multi-channel")
    parser.add_argument("--limit", type=int, default=None, help="cap number of images")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-grid", action="store_true", help="skip the overview grid per run")
    args = parser.parse_args()

    print(f"Sweeping {len(args.thresholds)} thresholds: "
          f"{[f'{t:.2f}' for t in args.thresholds]}")
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  images:     {args.images}")

    for t in args.thresholds:
        cmd = [
            sys.executable, str(detect_real),
            "--checkpoint", args.checkpoint,
            "--images", args.images,
            "--threshold", str(t),
            "--nms-dist", str(args.nms_dist),
            "--channel", str(args.channel),
            "--device", args.device,
        ]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        if args.no_grid:
            cmd += ["--no-grid"]
        print(f"\n=== threshold {t:.2f}  ->  results/thresh_{t:.2f}/ ===")
        subprocess.run(cmd, check=True)

    print("\nSweep complete. Compare the per-threshold folders under results/:")
    for t in args.thresholds:
        print(f"  results/thresh_{t:.2f}/")


if __name__ == "__main__":
    main()
