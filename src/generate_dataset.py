"""Generate a labeled synthetic GUV training dataset from the calibrated model.

    uv run python src/generate_dataset.py --config configs/dataset.yaml

For each image we run the calibrated forward simulator and save:
  - the image            -> images/<split>/<id>.{npy|png}
  - a label file listing the in-focus GUVs as (x, y, apparent_diameter)
                         -> labels/<split>/<id>.{json|csv}
plus a manifest.csv (one row per image) and a dataset_meta.json.

CRITICAL: only IN-FOCUS GUVs are labeled. The out-of-focus haze and the
saturated aggregates are distractors and are NEVER written to the labels -- the
detector must learn to ignore them. (`simulate_image`'s returned ground truth
already excludes them by construction; this script never invents extra labels.)

Under-constrained parameters are randomized PER IMAGE (see configs/dataset.yaml):
calibration constrains the physics but NOT the object population or the
degenerate gain/enf split, so those are sampled over documented ranges while the
constrained physics stays fixed at the calibrated values.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.forward_model import load_config, simulate_image  # noqa: E402


def _u(rng: np.random.Generator, pair) -> float:
    """Uniform sample from a [lo, hi] config pair."""
    lo, hi = float(pair[0]), float(pair[1])
    return float(rng.uniform(lo, hi))


def build_image_config(base_cfg: dict, rand_cfg: dict, rng: np.random.Generator) -> dict:
    """Build one image's simulator config: calibrated base + per-image randomized
    under-constrained params (see configs/dataset.yaml for the rationale)."""
    cfg = copy.deepcopy(base_cfg)

    # Object population.
    cfg["guvs"]["count"] = None  # use the (randomized) density via Poisson
    cfg["guvs"]["density"] = _u(rng, rand_cfg["guv_density"])
    cfg["guvs"]["sphere_diameter_log_mean"] = _u(rng, rand_cfg["size_log_mean"])
    cfg["guvs"]["sphere_diameter_log_sigma"] = _u(rng, rand_cfg["size_log_sigma"])

    # Focal-plane-cut fill factor (ring <-> disc continuum), never constrained.
    cfg["cut"]["axial_extent"] = _u(rng, rand_cfg["cut_axial_extent"])

    # Degenerate gain/enf: keep product ~ fitted, vary the split via enf.
    base_product = float(base_cfg["noise"]["gain"]) * float(base_cfg["noise"]["enf"])
    product = base_product * _u(rng, rand_cfg["product_jitter"])
    enf = _u(rng, rand_cfg["enf"])
    cfg["noise"]["enf"] = enf
    cfg["noise"]["gain"] = product / enf

    return cfg


# --------------------------------------------------------------------------- #
# Writers
# --------------------------------------------------------------------------- #
def _write_image(out_dir: Path, split: str, stem: str, image: np.ndarray, fmt: str) -> str:
    if fmt == "npy":
        rel = f"images/{split}/{stem}.npy"
        np.save(out_dir / rel, image.astype(np.float32))
    elif fmt == "png":
        from PIL import Image  # lazy: only needed for png output

        rel = f"images/{split}/{stem}.png"
        arr = np.clip(np.round(image), 0, 255).astype(np.uint8)
        Image.fromarray(arr).save(out_dir / rel)
    else:
        raise ValueError(f"unknown image_format: {fmt!r} (use 'npy' or 'png')")
    return rel


def _write_label(out_dir: Path, split: str, stem: str, truth: list, size: int,
                 image_rel: str, fmt: str) -> str:
    # truth is a list of (x, y, apparent_diameter) -- in-focus GUVs ONLY.
    guvs = [{"x": float(x), "y": float(y), "diameter": float(d)} for (x, y, d) in truth]
    if fmt == "json":
        rel = f"labels/{split}/{stem}.json"
        (out_dir / rel).write_text(
            json.dumps({"id": stem, "size": size, "image": image_rel, "guvs": guvs})
        )
    elif fmt == "csv":
        rel = f"labels/{split}/{stem}.csv"
        with open(out_dir / rel, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["x", "y", "diameter"])
            for g in guvs:
                w.writerow([g["x"], g["y"], g["diameter"]])
    else:
        raise ValueError(f"unknown label_format: {fmt!r} (use 'json' or 'csv')")
    return rel


def _write_manifest(out_dir: Path, rows: list) -> None:
    with open(out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "split", "image", "label", "n_guvs"])
        for r in rows:
            w.writerow([r["id"], r["split"], r["image"], r["label"], r["n_guvs"]])


def _write_meta(out_dir: Path, ds_cfg: dict, base_cfg: dict, n: int, n_val: int) -> None:
    meta = {
        "n_images": n,
        "n_train": n - n_val,
        "n_val": n_val,
        "seed": ds_cfg["dataset"]["seed"],
        "image_format": ds_cfg["output"]["image_format"],
        "label_format": ds_cfg["output"]["label_format"],
        "base_config": ds_cfg.get("base_config"),
        "randomized_params": ds_cfg["randomize"],
        "fixed_from_calibration": {
            "psf.sigma": base_cfg["psf"]["sigma"],
            "ring.brightness": base_cfg["ring"]["brightness"],
            "ring.thickness": base_cfg["ring"]["thickness"],
            "ring.rim_variation": base_cfg["ring"]["rim_variation"],
            "noise.offset": base_cfg["noise"]["offset"],
            "noise.read_noise": base_cfg["noise"]["read_noise"],
            "noise.optical_bg": base_cfg["noise"]["optical_bg"],
            "background.oof_amplitude": base_cfg["background"]["oof_amplitude"],
        },
        "note": (
            "Only in-focus GUVs are labeled; out-of-focus haze and saturated "
            "aggregates are distractors and are never labeled."
        ),
    }
    (out_dir / "dataset_meta.json").write_text(json.dumps(meta, indent=2))


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def generate_dataset(ds_cfg: dict, base_cfg: dict, out_dir, progress: bool = False) -> dict:
    """Generate the dataset described by `ds_cfg` using `base_cfg` as the
    calibrated simulator base. Returns a small summary dict."""
    out_dir = Path(out_dir)
    rand_cfg = ds_cfg["randomize"]
    n = int(ds_cfg["dataset"]["n_images"])
    val_ratio = float(ds_cfg["dataset"]["val_ratio"])
    seed = int(ds_cfg["dataset"]["seed"])
    img_fmt = ds_cfg["output"]["image_format"]
    lbl_fmt = ds_cfg["output"]["label_format"]
    size = int(base_cfg["image"]["size"])

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Reproducible, independent streams for the split assignment and per-image gen.
    split_ss, gen_ss = np.random.SeedSequence(seed).spawn(2)
    n_val = int(round(n * val_ratio))
    order = np.arange(n)
    np.random.default_rng(split_ss).shuffle(order)
    val_ids = set(order[:n_val].tolist())
    child_seeds = gen_ss.spawn(n)

    rows = []
    for i in range(n):
        rng = np.random.default_rng(child_seeds[i])
        cfg = build_image_config(base_cfg, rand_cfg, rng)
        image, truth = simulate_image(cfg, rng, return_truth=True)

        split = "val" if i in val_ids else "train"
        stem = f"{i:06d}"
        image_rel = _write_image(out_dir, split, stem, image, img_fmt)
        label_rel = _write_label(out_dir, split, stem, truth, size, image_rel, lbl_fmt)
        rows.append({"id": stem, "split": split, "image": image_rel,
                     "label": label_rel, "n_guvs": len(truth)})

        if progress and (i + 1) % 200 == 0:
            print(f"  {i + 1}/{n} images")

    _write_manifest(out_dir, rows)
    _write_meta(out_dir, ds_cfg, base_cfg, n, n_val)

    return {
        "out_dir": str(out_dir),
        "n_images": n,
        "n_train": n - n_val,
        "n_val": n_val,
        "total_labels": sum(r["n_guvs"] for r in rows),
    }


def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(repo_root / "configs" / "dataset.yaml"))
    args = parser.parse_args()

    with open(args.config) as f:
        ds_cfg = yaml.safe_load(f)

    base_path = repo_root / ds_cfg["base_config"]
    if not base_path.exists():
        raise FileNotFoundError(
            f"base_config not found: {base_path}\n"
            "Run calibration first (src/calibrate.py) to produce fitted_config.yaml."
        )
    base_cfg = load_config(base_path)
    out_dir = repo_root / ds_cfg["output"]["dir"]

    print(f"Generating {ds_cfg['dataset']['n_images']} images -> {out_dir}")
    print(f"  base (calibrated) config: {base_path}")
    summary = generate_dataset(ds_cfg, base_cfg, out_dir, progress=True)
    print(f"Done: {summary['n_train']} train + {summary['n_val']} val, "
          f"{summary['total_labels']} total GUV labels")
    print(f"  manifest: {out_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
