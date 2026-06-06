"""Render preview grids of synthetic GUV images to eyeball the simulator.

Produces a PNG you can compare against a real 561 nm field, showing all the
structural features the simulator models:

  - crowded full-pipeline panels: the ring/disc mix, the small-heavy size
    distribution, the diffuse out-of-focus haze, and the bright saturated
    aggregate distractors, all together;
  - a "ring vs disc" demo: the same sphere cut at the equator (ring) and off
    -equator (filled disc), so the focal-plane-cut behavior is explicit;
  - a "size sweep" panel: equatorial rings from d_min to d_max, to check the
    smallest rings still read as open annuli and the largest fit the frame.

Usage:
    uv run python scripts/preview_synthetic.py
    uv run python scripts/preview_synthetic.py --config configs/sim_default.yaml \
        --out previews/synthetic_preview.png --seed 0
"""

import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

# Allow running as a plain script (python scripts/...) without installing.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.forward_model import (  # noqa: E402
    GUV,
    _apply_noise,
    _render_clean,
    _render_oof_background,
    load_config,
    simulate_image,
)

# Decently-crowded density for the full-pipeline panels.
CROWDED_DENSITY = 40


def _normalize(img: np.ndarray) -> np.ndarray:
    """Min-max stretch to [0, 1] for display only (does not touch the sim)."""
    lo, hi = float(img.min()), float(img.max())
    if hi <= lo:
        return np.zeros_like(img)
    return (img - lo) / (hi - lo)


def _render_guv_list(guvs: list, cfg: dict, rng: np.random.Generator, with_bg: bool) -> np.ndarray:
    """Run a fixed GUV list through the full optical+noise pipeline (for demos)."""
    size = int(cfg["image"]["size"])
    clean = _render_clean(guvs, size, cfg["ring"], cfg["cut"], rng)
    in_focus = gaussian_filter(clean, sigma=cfg["psf"]["sigma"], mode="constant")
    oof = _render_oof_background(size, cfg["background"], rng) if with_bg else 0.0
    return _apply_noise(in_focus + oof, cfg["noise"], rng)


def _ring_vs_disc_demo(cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """Same sphere, equatorial cut (ring) vs off-equator cut (disc), side by side."""
    size = int(cfg["image"]["size"])
    sphere_d = 90.0
    R = sphere_d / 2.0
    quarter, threeq = size // 4, 3 * size // 4

    # Equatorial cut -> ring (apparent diameter == sphere diameter).
    ring = GUV(x=quarter, y=size // 2, apparent_diameter=sphere_d, sphere_diameter=sphere_d, cut_offset=0.0)
    # Off-equator cut -> smaller filled disc.
    h = 0.92 * R
    r_app = np.sqrt(R**2 - h**2)
    disc = GUV(x=threeq, y=size // 2, apparent_diameter=2 * r_app, sphere_diameter=sphere_d, cut_offset=h)

    demo_cfg = copy.deepcopy(cfg)
    demo_cfg["ring"]["brightness_jitter"] = 0.0
    return _render_guv_list([ring, disc], demo_cfg, rng, with_bg=False)


def _size_sweep(cfg: dict, rng: np.random.Generator, n: int = 9) -> np.ndarray:
    """Equatorial rings with apparent diameters evenly spanning d_min..d_max."""
    size = int(cfg["image"]["size"])
    d_min, d_max = cfg["guvs"]["d_min"], cfg["guvs"]["d_max"]
    diameters = np.linspace(d_min, d_max, n)

    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    xs = np.linspace(size / (cols + 1), size * cols / (cols + 1), cols)
    ys = np.linspace(size / (rows + 1), size * rows / (rows + 1), rows)

    guvs = []
    for i in range(n):
        d = float(diameters[i])
        guvs.append(
            GUV(
                x=xs[i % cols],
                y=ys[i // cols],
                apparent_diameter=d,
                sphere_diameter=d,  # equatorial cut: apparent == sphere
                cut_offset=0.0,
            )
        )
    sweep_cfg = copy.deepcopy(cfg)
    sweep_cfg["ring"]["brightness_jitter"] = 0.0
    return _render_guv_list(guvs, sweep_cfg, rng, with_bg=False)


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(repo_root / "configs" / "sim_default.yaml"))
    parser.add_argument("--out", default=str(repo_root / "previews" / "synthetic_preview.png"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    config = load_config(args.config)
    rng = np.random.default_rng(args.seed)

    panels = []  # (title, image)

    # Three crowded full-pipeline panels: ring/disc mix + small-heavy sizes + haze.
    crowded = copy.deepcopy(config)
    crowded["guvs"]["density"] = CROWDED_DENSITY
    crowded["guvs"]["count"] = None
    for k in range(3):
        panels.append((f"crowded field #{k + 1}\n(rings+discs+haze+aggregates)", simulate_image(crowded, rng)))

    # Focal-plane-cut demo and size sweep.
    panels.append(("ring vs disc\n(equator | off-equator cut)", _ring_vs_disc_demo(config, rng)))
    d_min, d_max = config["guvs"]["d_min"], config["guvs"]["d_max"]
    panels.append((f"size sweep: d={d_min}..{d_max} px\n(equatorial rings)", _size_sweep(config, rng)))

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 4.5))
    for ax, (title, img) in zip(axes, panels):
        ax.imshow(_normalize(img), cmap="gray")
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    fig.suptitle("Synthetic GUV previews (display-normalized)", fontsize=14)
    fig.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved preview to {out_path}")


if __name__ == "__main__":
    main()
