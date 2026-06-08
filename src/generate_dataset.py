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
import multiprocessing as mp
import os
import sys
from pathlib import Path

import yaml

# NOTE: numpy and the forward model (which pulls in numpy/scipy) are imported
# LAZILY inside the functions that use them -- NOT at module top level. Under the
# 'spawn' multiprocessing start method each worker re-imports this module before
# its initializer runs; importing numpy at top level would happen BEFORE the
# initializer sets OMP/MKL/OPENBLAS_NUM_THREADS=1, so every worker's BLAS would
# spin up its own thread pool and oversubscribe the cores (slower, not faster).
# Keeping these imports lazy lets the single-thread env vars take effect first.

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _u(rng, pair) -> float:
    """Uniform sample from a [lo, hi] config pair."""
    lo, hi = float(pair[0]), float(pair[1])
    return float(rng.uniform(lo, hi))


def build_image_config(base_cfg: dict, rand_cfg: dict, size_cfg: dict, rng,
                       place_cfg: dict | None = None) -> dict:
    """Build one image's simulator config: calibrated base + per-image randomized
    under-constrained params + the fixed sphere-diameter mixture + the fixed
    soft-exclusion placement knobs (see configs/dataset.yaml for the rationale)."""
    cfg = copy.deepcopy(base_cfg)

    # Object population.
    cfg["guvs"]["count"] = None  # use the (randomized) density via Poisson
    cfg["guvs"]["density"] = _u(rng, rand_cfg["guv_density"])

    # Sphere-diameter SIZE distribution: fixed two-component lognormal mixture
    # (a deliberate population choice, NOT calibrated physics). Copy the five
    # knobs straight from size_cfg into the simulator's guv config.
    cfg["guvs"]["size_dist"] = size_cfg.get("size_dist", "mixture")
    for k in ("small_fraction", "small_log_mean", "small_log_sigma",
              "large_log_mean", "large_log_sigma"):
        cfg["guvs"][k] = float(size_cfg[k])

    # Soft-exclusion PLACEMENT knobs (overlap decoupled from density). Fixed, not
    # randomized; injected here so they override the base config for generation.
    if place_cfg is not None:
        for k in ("min_separation_factor", "allowed_overlap_fraction",
                  "placement_max_attempts"):
            if k in place_cfg:
                cfg["guvs"][k] = place_cfg[k]

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
def _write_image(out_dir: Path, split: str, stem: str, image, fmt: str) -> str:
    import numpy as np  # lazy (see module header)

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
        "size_distribution": ds_cfg.get("size_distribution"),
        "placement": ds_cfg.get("placement"),
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
# Parallel workers
# --------------------------------------------------------------------------- #
# Each worker holds the (read-only) shared generation context in this module
# global, populated ONCE per worker by the Pool initializer. This avoids
# re-pickling the base config for every image.
_CTX: dict = {}


def _worker_init(ctx: dict) -> None:
    """Pool worker initializer. Runs ONCE per fresh ('spawn') interpreter, BEFORE
    any task -- and crucially before numpy is imported in the task -- so we pin
    every worker's numpy/BLAS to a single thread. Without this, N workers each
    spawn ~N_cores BLAS threads and oversubscribe the machine (slower, not
    faster). Mirrors the single-threaded-worker-under-spawn pattern used in the
    calibration study runner."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    _CTX.update(ctx)


def _generate_one(task) -> dict:
    """Generate, label and write ONE image. `task` is `(i, child_seed)` where `i`
    is the global image index and `child_seed` is the index-derived SeedSequence
    (see generate_dataset). Heavy imports are LAZY so the initializer's
    single-thread env vars are in effect before numpy/BLAS load."""
    import numpy as np  # lazy (see module header)

    from src.forward_model import simulate_image  # lazy: pulls in numpy/scipy

    i, child_seed = task
    ctx = _CTX
    # Seed PER IMAGE INDEX, not per worker: child_seed depends only on (seed, i),
    # so the content of image i is identical regardless of how images are
    # distributed across workers -- two workers can never collide or duplicate.
    rng = np.random.default_rng(child_seed)
    cfg = build_image_config(ctx["base_cfg"], ctx["rand_cfg"], ctx["size_cfg"], rng,
                             place_cfg=ctx["place_cfg"])
    image, truth = simulate_image(cfg, rng, return_truth=True)

    split = "val" if i in ctx["val_ids"] else "train"
    stem = f"{i:06d}"
    image_rel = _write_image(ctx["out_dir"], split, stem, image, ctx["img_fmt"])
    label_rel = _write_label(ctx["out_dir"], split, stem, truth, ctx["size"],
                             image_rel, ctx["lbl_fmt"])
    return {"id": stem, "split": split, "image": image_rel,
            "label": label_rel, "n_guvs": len(truth)}


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def generate_dataset(ds_cfg: dict, base_cfg: dict, out_dir, progress: bool = False,
                     n_workers: int = 1) -> dict:
    """Generate the dataset described by `ds_cfg` using `base_cfg` as the
    calibrated simulator base. Returns a small summary dict.

    `n_workers` controls parallelism: <=1 runs in-process (used by tests and for
    debugging); >1 uses a 'spawn' multiprocessing.Pool of single-threaded
    workers. Parallelism changes ONLY speed and output order -- per-image content
    and the train/val split are identical to the serial path, because both seed
    every image by its global index (not by worker)."""
    import numpy as np  # lazy (see module header)

    out_dir = Path(out_dir)
    rand_cfg = ds_cfg["randomize"]
    size_cfg = ds_cfg["size_distribution"]
    place_cfg = ds_cfg.get("placement")  # soft-exclusion placement knobs (optional)
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
    # child_seeds[i] is derived purely from (seed, i), so the seeding is identical
    # in the serial and parallel paths and across any worker count.
    split_ss, gen_ss = np.random.SeedSequence(seed).spawn(2)
    n_val = int(round(n * val_ratio))
    order = np.arange(n)
    np.random.default_rng(split_ss).shuffle(order)
    val_ids = set(order[:n_val].tolist())
    child_seeds = gen_ss.spawn(n)

    ctx = {
        "base_cfg": base_cfg, "rand_cfg": rand_cfg, "size_cfg": size_cfg,
        "place_cfg": place_cfg,
        "out_dir": out_dir, "img_fmt": img_fmt, "lbl_fmt": lbl_fmt,
        "size": size, "val_ids": val_ids,
    }
    tasks = [(i, child_seeds[i]) for i in range(n)]

    if n_workers and n_workers > 1:
        # Parallel: fresh 'spawn' interpreters, single-threaded BLAS per worker.
        rows = []
        ctx_proc = mp.get_context("spawn")
        with ctx_proc.Pool(processes=int(n_workers), initializer=_worker_init,
                           initargs=(ctx,)) as pool:
            for done, row in enumerate(pool.imap_unordered(_generate_one, tasks), 1):
                rows.append(row)
                if progress and done % 200 == 0:
                    print(f"  {done}/{n} images")
        rows.sort(key=lambda r: r["id"])  # deterministic manifest order
    else:
        # Serial (in-process): identical content, no spawn overhead.
        _worker_init(ctx)
        rows = []
        for done, task in enumerate(tasks, 1):
            rows.append(_generate_one(task))
            if progress and done % 200 == 0:
                print(f"  {done}/{n} images")

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
    parser.add_argument(
        "--n-workers", type=int, default=os.cpu_count(),
        help="parallel worker processes (default: all CPU cores). Generation is "
             "embarrassingly parallel; each worker is a single-threaded 'spawn' "
             "interpreter. Use 1 to run serially.",
    )
    args = parser.parse_args()

    from src.forward_model import load_config  # lazy (see module header)

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
    print(f"  workers: {args.n_workers}")
    summary = generate_dataset(ds_cfg, base_cfg, out_dir, progress=True,
                               n_workers=args.n_workers)
    print(f"Done: {summary['n_train']} train + {summary['n_val']} val, "
          f"{summary['total_labels']} total GUV labels")
    print(f"  manifest: {out_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
