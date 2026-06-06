"""Torch Dataset for the synthetic GUV data + CenterNet target construction.

Loads uint8 PNG images (normalized to [0, 1] via the SHARED `src.normalize`
path, identical to inference) and their labels, and builds CenterNet-style
training targets:

  - center heatmap : a Gaussian blob at each labeled GUV center, with sigma
                     scaled to the object's radius; trained with focal loss.
  - radius map     : the labeled radius written at each center pixel; trained
                     with L1, supervised ONLY at true centers (via reg_mask).

Targets are built at (near-)full resolution -- `down_ratio` defaults to 1 -- so
that close centers in crowded fields stay separable (no heavy downsampling).

Only IN-FOCUS GUVs are in the labels; the out-of-focus haze and the saturated
aggregates are never labeled, so the network is trained to ignore them.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.normalize import load_normalized


def _draw_gaussian(heatmap: np.ndarray, cx: int, cy: int, sigma: float) -> None:
    """Render a Gaussian peak (max-combined) of value 1.0 at center into heatmap."""
    h, w = heatmap.shape
    radius = int(max(1, round(3 * sigma)))  # cover ~3 sigma
    x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.ogrid[y0:y1, x0:x1]
    g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2.0 * sigma**2))
    region = heatmap[y0:y1, x0:x1]
    np.maximum(region, g, out=region)  # overlapping blobs take the max


def build_targets(
    guvs: list,
    size: int,
    down_ratio: int = 1,
    sigma_scale: float = 0.15,
    sigma_min: float = 1.5,
    sigma_max: float = 8.0,
) -> tuple:
    """Build (heatmap, radius_map, reg_mask), each (out_h, out_w) float32.

    `guvs` is a list of (x, y, diameter) in input pixels. Centers and the heatmap
    live at output resolution (size // down_ratio); the regressed radius is kept
    in INPUT pixels so detect.py reports radii directly comparable to the labels.
    """
    out = size // down_ratio
    heatmap = np.zeros((out, out), dtype=np.float32)
    radius_map = np.zeros((out, out), dtype=np.float32)
    reg_mask = np.zeros((out, out), dtype=np.float32)

    for x, y, diameter in guvs:
        radius = float(diameter) / 2.0
        cx = int(round(float(x) / down_ratio))
        cy = int(round(float(y) / down_ratio))
        if not (0 <= cx < out and 0 <= cy < out):
            continue  # center off-frame (clipped GUV) -> not a trainable center
        sigma = float(np.clip((radius / down_ratio) * sigma_scale, sigma_min, sigma_max))
        _draw_gaussian(heatmap, cx, cy, sigma)
        heatmap[cy, cx] = 1.0          # exact center is a positive
        radius_map[cy, cx] = radius    # radius in INPUT pixels
        reg_mask[cy, cx] = 1.0

    return heatmap, radius_map, reg_mask


def _load_labels(path: Path) -> list:
    """Load a label file (json or csv) -> list of (x, y, diameter)."""
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        return [(g["x"], g["y"], g["diameter"]) for g in data["guvs"]]
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append((float(r["x"]), float(r["y"]), float(r["diameter"])))
    return rows


def read_manifest(root: Path, split: str) -> list:
    """Return manifest rows (dicts) for a split, in manifest order."""
    with open(Path(root) / "manifest.csv", newline="") as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


class GUVDataset(Dataset):
    """Synthetic GUV dataset producing (image, targets) for training.

    Each item is a dict:
        image      : float32 (1, H, W)        normalized to [0, 1]
        heatmap    : float32 (1, oH, oW)       center heatmap (focal-loss target)
        radius_map : float32 (1, oH, oW)       radius (input px) at centers
        reg_mask   : float32 (1, oH, oW)       1 at centers (L1 supervision mask)
        n_guvs     : int                       number of labeled GUVs
    """

    def __init__(
        self,
        root,
        split: str,
        down_ratio: int = 1,
        sigma_scale: float = 0.15,
        sigma_min: float = 1.5,
        sigma_max: float = 8.0,
        limit: int | None = None,
    ):
        self.root = Path(root)
        self.rows = read_manifest(self.root, split)
        if limit is not None:
            self.rows = self.rows[: int(limit)]
        self.down_ratio = down_ratio
        self.sigma_scale = sigma_scale
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        img = load_normalized(self.root / row["image"])  # (H, W) float32 [0,1]
        size = img.shape[0]
        guvs = _load_labels(self.root / row["label"])
        heatmap, radius_map, reg_mask = build_targets(
            guvs, size, self.down_ratio, self.sigma_scale, self.sigma_min, self.sigma_max
        )
        return {
            "image": torch.from_numpy(img)[None],          # (1,H,W)
            "heatmap": torch.from_numpy(heatmap)[None],     # (1,oH,oW)
            "radius_map": torch.from_numpy(radius_map)[None],
            "reg_mask": torch.from_numpy(reg_mask)[None],
            "n_guvs": len(guvs),
        }
