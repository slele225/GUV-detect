"""Evaluate the detector on the held-out synthetic val set.

    uv run python src/evaluate.py --config configs/eval.yaml

Reports overall precision / recall / F1 and radius MAE, AND precision/recall
binned by crowding (number of GUVs per image) -- the reference curve that stands
in for a Hough baseline. Matching is greedy by center distance within
`tol_frac * gt_radius` (see src/metrics.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import _load_labels, read_manifest  # noqa: E402
from src.detect import detect_image, load_model  # noqa: E402
from src.metrics import aggregate, match_detections, precision_recall_f1  # noqa: E402
from src.normalize import load_normalized  # noqa: E402


def _bin_label(n: int, edges: list) -> str:
    """Map a GUV count to a crowding-bin label using right-open edges."""
    for i in range(len(edges) - 1):
        if edges[i] <= n < edges[i + 1]:
            return f"{edges[i]}-{edges[i + 1] - 1}"
    return f"{edges[-1]}+"


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(repo_root / "configs" / "eval.yaml"))
    parser.add_argument("--device", default=None, help="override device (e.g. cuda, cpu)")
    parser.add_argument("--checkpoint", default=None, help="override checkpoint path")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.checkpoint is not None:
        cfg["checkpoint"] = args.checkpoint

    device = args.device if args.device is not None else cfg.get("device", "cpu")
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device = "cpu"

    root = repo_root / cfg["data"]["root"]
    rows = read_manifest(root, cfg["data"].get("split", "val"))
    limit = cfg["data"].get("limit")
    if limit is not None:
        rows = rows[: int(limit)]

    model = load_model(repo_root / cfg["checkpoint"], device=device)
    det_cfg = {
        "threshold": cfg["detect"]["threshold"],
        "nms_dist": cfg["detect"]["nms_dist"],
        "down_ratio": cfg["detect"]["down_ratio"],
    }
    tol_frac = cfg["match"]["tol_frac"]
    edges = cfg["crowding_bins"]

    per_image = []
    by_bin = {}
    for r in rows:
        image = load_normalized(root / r["image"])
        gts = _load_labels(root / r["label"])
        preds = detect_image(model, image, det_cfg, device=device)
        res = match_detections(preds, gts, tol_frac=tol_frac)
        per_image.append(res)

        label = _bin_label(len(gts), edges)
        by_bin.setdefault(label, []).append(res)

    overall = aggregate(per_image)
    print(f"\n=== Overall (n={len(rows)} val images, tol={tol_frac}*radius) ===")
    print(f"precision={overall['precision']:.3f}  recall={overall['recall']:.3f}  "
          f"F1={overall['f1']:.3f}  radius_MAE={overall['radius_mae']:.2f}px")
    print(f"  (TP={overall['tp']} FP={overall['fp']} FN={overall['fn']})")

    print("\n=== Precision / recall vs. crowding (GUVs per image) ===")
    print(f"{'bin':>10} {'images':>7} {'precision':>10} {'recall':>8} {'F1':>6} {'radMAE':>7}")

    def _sort_key(lbl):
        return int(lbl.split("-")[0].rstrip("+"))

    for label in sorted(by_bin, key=_sort_key):
        agg = aggregate(by_bin[label])
        print(f"{label:>10} {len(by_bin[label]):>7} {agg['precision']:>10.3f} "
              f"{agg['recall']:>8.3f} {agg['f1']:>6.3f} {agg['radius_mae']:>7.2f}")


if __name__ == "__main__":
    main()
