"""Inference: run the GUV detector on an image and decode (x, y, radius) circles.

Pipeline (identical for synthetic and real uint8 images):
    load image via the SHARED `src.normalize.load_normalized` -> [0, 1]
    -> model forward -> heatmap + radius
    -> peak-pick local maxima above a threshold
    -> read the radius at each peak
    -> NMS by center distance
    -> list of (x, y, radius, score)

`decode()` is reused by train.py (val visualizations) and evaluate.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model import GUVNet  # noqa: E402
from src.normalize import load_normalized  # noqa: E402


def _local_maxima(hm: torch.Tensor, kernel: int = 3) -> torch.Tensor:
    """Keep only pixels that are the max in their kernel x kernel neighborhood."""
    pad = kernel // 2
    pooled = F.max_pool2d(hm, kernel, stride=1, padding=pad)
    return hm * (pooled == hm).float()


def nms_by_distance(dets: np.ndarray, min_dist: float) -> np.ndarray:
    """Greedy NMS: keep highest-score detections, drop others within min_dist.

    `dets` is (N, 4) rows (x, y, radius, score). Returns the kept rows.
    """
    if len(dets) == 0:
        return dets
    order = np.argsort(-dets[:, 3])  # by score desc
    kept = []
    taken = np.zeros(len(dets), dtype=bool)
    for i in order:
        if taken[i]:
            continue
        kept.append(i)
        dx = dets[:, 0] - dets[i, 0]
        dy = dets[:, 1] - dets[i, 1]
        close = (dx * dx + dy * dy) <= (min_dist * min_dist)
        taken |= close
    return dets[sorted(kept)]


def decode(
    hm: torch.Tensor,
    radius: torch.Tensor,
    threshold: float = 0.3,
    nms_dist: float = 6.0,
    down_ratio: int = 1,
    max_dets: int = 1000,
) -> np.ndarray:
    """Decode one heatmap+radius map into detections (x, y, radius, score).

    `hm`/`radius` are (1, H, W) or (H, W) tensors for a SINGLE image.
    """
    if hm.dim() == 3:
        hm = hm[0]
    if radius.dim() == 3:
        radius = radius[0]

    peaks = _local_maxima(hm[None, None])[0, 0]
    ys, xs = torch.where(peaks >= threshold)
    if len(xs) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    scores = peaks[ys, xs]
    rad = radius[ys, xs]
    x_img = xs.float() * down_ratio
    y_img = ys.float() * down_ratio
    dets = torch.stack([x_img, y_img, rad, scores], dim=1).detach().cpu().numpy()

    # Keep the strongest before NMS (cheap guard for pathological heatmaps).
    if len(dets) > max_dets:
        dets = dets[np.argsort(-dets[:, 3])[:max_dets]]
    return nms_by_distance(dets, nms_dist)


@torch.no_grad()
def detect_image(model, image_unit: np.ndarray, cfg: dict, device: str = "cpu") -> np.ndarray:
    """Run the model on a normalized [0,1] (H,W) image and return detections."""
    model.eval()
    x = torch.from_numpy(image_unit)[None, None].to(device)
    out = model(x)
    return decode(
        out["hm"][0], out["radius"][0],
        threshold=cfg.get("threshold", 0.3),
        nms_dist=cfg.get("nms_dist", 6.0),
        down_ratio=cfg.get("down_ratio", 1),
        max_dets=cfg.get("max_dets", 1000),
    )


def load_model(checkpoint: str | Path, device: str = "cpu") -> GUVNet:
    ckpt = torch.load(checkpoint, map_location=device)
    m = ckpt.get("model_cfg", {})
    model = GUVNet(
        in_ch=1,
        base=m.get("base", 32),
        depth=m.get("depth", 4),
        out_stride=m.get("out_stride", 1),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Detect GUVs in an image.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True, help="PNG or TIFF (real or synthetic)")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--nms-dist", type=float, default=6.0)
    parser.add_argument("--down-ratio", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=None, help="optional JSON output path")
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device)
    image = load_normalized(args.image)  # SAME normalization as training
    cfg = {"threshold": args.threshold, "nms_dist": args.nms_dist, "down_ratio": args.down_ratio}
    dets = detect_image(model, image, cfg, device=args.device)

    result = [{"x": float(d[0]), "y": float(d[1]), "radius": float(d[2]), "score": float(d[3])}
              for d in dets]
    print(f"{len(result)} GUVs detected")
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")
    else:
        for r in result:
            print(f"  x={r['x']:.1f} y={r['y']:.1f} r={r['radius']:.1f} score={r['score']:.3f}")


if __name__ == "__main__":
    main()
