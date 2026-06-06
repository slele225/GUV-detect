"""Overlay ground-truth labels on generated images to verify the dataset.

    uv run python scripts/preview_dataset.py
    uv run python scripts/preview_dataset.py --config configs/dataset.yaml \
        --n 9 --split train --out dataset/preview_labels.png

Loads a handful of generated images and draws a circle at each labeled
(x, y) with the labeled diameter. Use it to check that labels land on the
in-focus rings/discs -- and that the out-of-focus haze and the bright saturated
aggregates are correctly NOT circled (they must stay unlabeled).
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.patches import Circle


def _load_image(out_dir: Path, rel: str) -> np.ndarray:
    if rel.endswith(".npy"):
        return np.load(out_dir / rel)
    from PIL import Image

    return np.asarray(Image.open(out_dir / rel)).astype(np.float64)


def _load_labels(out_dir: Path, rel: str) -> list:
    if rel.endswith(".json"):
        data = json.loads((out_dir / rel).read_text())
        return [(g["x"], g["y"], g["diameter"]) for g in data["guvs"]]
    rows = []
    with open(out_dir / rel, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((float(r["x"]), float(r["y"]), float(r["diameter"])))
    return rows


def _normalize(img: np.ndarray) -> np.ndarray:
    lo, hi = float(img.min()), float(img.max())
    return (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(repo_root / "configs" / "dataset.yaml"))
    parser.add_argument("--n", type=int, default=9)
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        ds_cfg = yaml.safe_load(f)
    out_dir = repo_root / ds_cfg["output"]["dir"]

    manifest = out_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"no manifest at {manifest}; run generate_dataset.py first")
    with open(manifest, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["split"] == args.split]
    if not rows:
        raise ValueError(f"no images for split {args.split!r} in {manifest}")
    rows = rows[: args.n]

    cols = int(np.ceil(np.sqrt(len(rows))))
    grid_rows = int(np.ceil(len(rows) / cols))
    fig, axes = plt.subplots(grid_rows, cols, figsize=(4 * cols, 4 * grid_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, r in zip(axes, rows):
        img = _load_image(out_dir, r["image"])
        labels = _load_labels(out_dir, r["label"])
        ax.imshow(_normalize(img), cmap="gray")
        for x, y, d in labels:
            ax.add_patch(Circle((x, y), d / 2.0, fill=False, edgecolor="red", linewidth=1.0))
        ax.set_title(f"{r['id']}: {len(labels)} labeled", fontsize=10)
        ax.axis("off")
    for ax in axes[len(rows):]:
        ax.axis("off")

    fig.suptitle("Dataset labels (red = in-focus GUVs; haze/aggregates must stay unlabeled)",
                 fontsize=13)
    fig.tight_layout()

    out_path = Path(args.out) if args.out else out_dir / "preview_labels.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved label preview to {out_path}")


if __name__ == "__main__":
    main()
