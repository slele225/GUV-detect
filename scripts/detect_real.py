"""Run the trained detector on REAL images and save visual comparisons.

This is the qualitative real-image transfer check -- there is NO ground truth, so
we just eyeball whether the synthetic-trained detector fires on real rings/discs
and ignores haze/aggregates.

For each real image it loads via the SAME shared [0,1] normalization used in
training and detect.py (src.normalize.load_normalized), runs the identical
inference pipeline (heatmap -> peak-pick -> NMS -> (x, y, radius)), and saves a
3-panel PNG: raw image | image + predicted circles/centers | predicted heatmap.

    uv run python scripts/detect_real.py --checkpoint runs/guvnet/best.pt
    uv run python scripts/detect_real.py --images data --threshold 0.4 --nms-dist 8

Tune --threshold / --nms-dist here to inspect transfer WITHOUT retraining.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.detect import decode, load_model  # noqa: E402
from src.normalize import load_normalized  # noqa: E402

IMAGE_EXTS = ("*.tif", "*.tiff", "*.png", "*.jpg", "*.jpeg")


def _stretch(img: np.ndarray, lo: float = 1.0, hi: float = 99.5) -> np.ndarray:
    """Percentile contrast stretch for DISPLAY only (detection uses raw [0,1])."""
    a, b = np.percentile(img, [lo, hi])
    if b <= a:
        return np.zeros_like(img)
    return np.clip((img - a) / (b - a), 0.0, 1.0)


@torch.no_grad()
def predict(model, image_unit: np.ndarray, threshold: float, nms_dist: float,
            down_ratio: int, device: str):
    """Run the detect.py pipeline; return (detections (N,4), heatmap (h,w))."""
    x = torch.from_numpy(image_unit)[None, None].to(device)
    out = model(x)
    hm = out["hm"][0, 0].cpu().numpy()
    dets = decode(out["hm"][0], out["radius"][0], threshold=threshold,
                  nms_dist=nms_dist, down_ratio=down_ratio)
    return dets, hm


def save_panel(image_unit, dets, hm, title, out_path):
    """Save the 3-panel comparison PNG for one image."""
    disp = _stretch(image_unit)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.4))

    axes[0].imshow(disp, cmap="gray")
    axes[0].set_title("real image")
    axes[0].axis("off")

    axes[1].imshow(disp, cmap="gray")
    for d in dets:
        x, y, r = float(d[0]), float(d[1]), float(d[2])
        axes[1].add_patch(Circle((x, y), r, fill=False, edgecolor="red", lw=1.2))
        axes[1].plot(x, y, "+", color="cyan", markersize=5, markeredgewidth=1.0)
    axes[1].set_title(f"detections: {len(dets)}")
    axes[1].axis("off")

    im = axes[2].imshow(hm, cmap="magma", vmin=0.0, vmax=1.0)
    axes[2].set_title("center heatmap")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def save_grid(items, out_path):
    """Overview grid of the overlay panels (image + detections) for all images."""
    n = len(items)
    if n == 0:
        return
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, (name, image_unit, dets, _hm) in zip(axes, items):
        ax.imshow(_stretch(image_unit), cmap="gray")
        for d in dets:
            ax.add_patch(Circle((float(d[0]), float(d[1])), float(d[2]),
                                fill=False, edgecolor="red", lw=0.9))
        ax.set_title(f"{name}: {len(dets)}", fontsize=9)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Real-image detections (qualitative transfer check)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=str(repo_root / "runs" / "guvnet" / "best.pt"))
    parser.add_argument("--images", default=str(repo_root / "data"),
                        help="folder of real images (default: data/)")
    parser.add_argument("--out", default=str(repo_root / "detect_real_out"))
    parser.add_argument("--threshold", type=float, default=0.3, help="heatmap peak threshold")
    parser.add_argument("--nms-dist", type=float, default=6.0, help="NMS center distance (px)")
    parser.add_argument("--channel", type=int, default=1, help="lipid channel if multi-channel")
    parser.add_argument("--limit", type=int, default=None, help="cap number of images")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-grid", action="store_true", help="skip the overview grid")
    args = parser.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available; using CPU.")
        device = "cpu"

    images_dir = Path(args.images)
    paths = sorted(p for ext in IMAGE_EXTS for p in images_dir.glob(ext))
    if not paths:
        raise FileNotFoundError(f"no images ({', '.join(IMAGE_EXTS)}) found in {images_dir}")
    if args.limit is not None:
        paths = paths[: args.limit]

    model = load_model(args.checkpoint, device=device)
    down_ratio = getattr(model, "out_stride", 1)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Detecting on {len(paths)} real images (threshold={args.threshold}, "
          f"nms_dist={args.nms_dist}, down_ratio={down_ratio}) -> {out_dir}")

    grid_items = []
    for p in paths:
        image_unit = load_normalized(p, channel=args.channel)  # SAME normalization as training
        dets, hm = predict(model, image_unit, args.threshold, args.nms_dist, down_ratio, device)
        title = f"{p.name}  -  {len(dets)} detections"
        save_panel(image_unit, dets, hm, title, out_dir / f"{p.stem}.png")
        grid_items.append((p.name, image_unit, dets, hm))
        print(f"  {p.name}: {len(dets)} detections")

    if not args.no_grid:
        save_grid(grid_items, out_dir / "_grid.png")
        print(f"  grid: {out_dir / '_grid.png'}")
    print("done.")


if __name__ == "__main__":
    main()
