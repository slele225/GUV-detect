"""Detection-free moment-matching calibration of the GUV forward simulator.

Approach (lipid-channel only): reduce both the real 561 nm images and batches of
simulated images to global summary statistics (src/statistics.py), then use
Optuna to minimize a config-driven weighted discrepancy (src/discrepancy.py)
between them. No detector and no per-object labels are involved -- we only match
the *statistics* of the lipid channel.

WHAT OPTUNA FITS vs. WHAT STAYS PINNED
    Global image statistics constrain brightness level, the bright tail, and
    spatial texture -- but they CANNOT constrain the focal-plane-cut geometry:
    the ring-vs-disc ratio is a minority effect on global moments. So the cut
    parameters (cut.axial_extent, cut.offset_max_frac) are PINNED at their
    hand-tuned values and never fitted here. Optuna fits only what the lipid
    statistics actually constrain -- declared in the calibration config's `fit`
    block (PSF sigma, ring brightness, the noise model, in-focus density, the
    lognormal size params, and the haze/aggregate amplitudes). See
    configs/calibration.yaml for the full list and rationale.

    Degeneracy note: gain and enf are degenerate (only their product is
    constrained by intensity stats). They are still fitted to a point here, but
    downstream synthetic-data GENERATION should randomize over them within their
    ranges rather than trusting the point estimate.

Run:
    uv run python src/calibrate.py --config configs/calibration_smoke.yaml   # quick
    uv run python src/calibrate.py --config configs/calibration.yaml         # real
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import tifffile
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.discrepancy import discrepancy  # noqa: E402
from src.forward_model import load_config, simulate_image  # noqa: E402
from src.statistics import average_statistics, compute_statistics  # noqa: E402


# --------------------------------------------------------------------------- #
# Config-path helpers (dotted keys -> nested dict)
# --------------------------------------------------------------------------- #
def _set_by_path(cfg: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node[k]
    node[keys[-1]] = value


# --------------------------------------------------------------------------- #
# Real-image loading (561 nm lipid channel)
# --------------------------------------------------------------------------- #
def _extract_lipid_channel(arr: np.ndarray, channel: int) -> np.ndarray:
    """Pick the lipid channel from a real image.

    Files in data/ are single-channel 512x512 here, but stay robust to 3-channel
    stacks: if 3D, take `channel` along whichever axis looks like channels.
    """
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        # channel-first (C, H, W) if a small leading axis, else channel-last.
        if arr.shape[0] <= 4:
            return arr[channel]
        if arr.shape[-1] <= 4:
            return arr[..., channel]
    raise ValueError(f"unsupported real-image shape: {arr.shape}")


def _center_crop(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape
    if h == size and w == size:
        return img
    top = max((h - size) // 2, 0)
    left = max((w - size) // 2, 0)
    return img[top : top + size, left : left + size]


def load_real_images(data_cfg: dict) -> list:
    """Load real lipid-channel images normalized to [0, 1]."""
    pattern = str(Path(data_cfg["dir"]) / data_cfg.get("glob", "*.tif*"))
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no real images matched {pattern}")

    channel = int(data_cfg.get("channel", 1))
    size = int(data_cfg.get("size", 512))
    crop = data_cfg.get("crop", "center")
    max_value = float(data_cfg.get("max_value", 255))
    limit = data_cfg.get("limit")
    if limit is not None:
        paths = paths[: int(limit)]

    images = []
    for p in paths:
        raw = _extract_lipid_channel(tifffile.imread(p), channel).astype(np.float64)
        if crop == "center":
            raw = _center_crop(raw, size)
        images.append(np.clip(raw / max_value, 0.0, 1.0))
    return images


# --------------------------------------------------------------------------- #
# Statistics helpers
# --------------------------------------------------------------------------- #
def _sim_to_unit(image: np.ndarray, base_cfg: dict) -> np.ndarray:
    """Normalize a simulated image (counts, clipped) to [0, 1] using sensor max."""
    sensor = base_cfg.get("sensor") or {}
    max_value = sensor.get("max_value", 255)
    return np.clip(image / float(max_value), 0.0, 1.0)


def _simulated_stats(sim_cfg: dict, stats_cfg: dict, n_images: int, rng: np.random.Generator) -> dict:
    per_image = []
    for _ in range(n_images):
        img = simulate_image(sim_cfg, rng)
        per_image.append(compute_statistics(_sim_to_unit(img, sim_cfg), stats_cfg))
    return average_statistics(per_image)


# --------------------------------------------------------------------------- #
# Optuna objective
# --------------------------------------------------------------------------- #
def _suggest_params(trial: optuna.Trial, fit_cfg: dict) -> dict:
    params = {}
    for dotted, spec in fit_cfg.items():
        params[dotted] = trial.suggest_float(
            dotted, float(spec["low"]), float(spec["high"]), log=bool(spec.get("log", False))
        )
    return params


def _build_sim_config(base_cfg: dict, params: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    for dotted, value in params.items():
        _set_by_path(cfg, dotted, value)
    return cfg


def run_calibration(cal_cfg: dict, base_cfg: dict, real_stats: dict):
    fit_cfg = cal_cfg["fit"]
    disc_cfg = cal_cfg["discrepancy"]
    stats_cfg = cal_cfg.get("statistics", {})
    opt_cfg = cal_cfg["optuna"]

    n_trials = int(opt_cfg["n_trials"])
    n_sim = int(opt_cfg.get("n_sim_images", 6))
    base_seed = int(opt_cfg.get("seed", 0))

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial, fit_cfg)
        sim_cfg = _build_sim_config(base_cfg, params)
        rng = np.random.default_rng(base_seed + 1000 + trial.number)
        sim_stats = _simulated_stats(sim_cfg, stats_cfg, n_sim, rng)
        total, breakdown = discrepancy(real_stats, sim_stats, disc_cfg)
        for k, v in breakdown.items():
            trial.set_user_attr(f"term_{k}", v)
        return total

    sampler = optuna.samplers.TPESampler(seed=base_seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study


# --------------------------------------------------------------------------- #
# Outputs: fitted params JSON + comparison plots
# --------------------------------------------------------------------------- #
def _write_outputs(cal_cfg, base_cfg, study, real_stats, real_images, stats_cfg):
    out_cfg = cal_cfg.get("output", {})
    out_dir = Path(out_cfg.get("dir", "calibration_out"))
    out_dir.mkdir(parents=True, exist_ok=True)

    best_params = study.best_trial.params
    best_sim_cfg = _build_sim_config(base_cfg, best_params)

    # Fitted params JSON (overrides + metadata).
    fitted_json = out_dir / out_cfg.get("fitted_json", "fitted_params.json")
    payload = {
        "fitted_params": best_params,
        "best_value": study.best_value,
        "term_breakdown": {
            k.replace("term_", ""): v
            for k, v in study.best_trial.user_attrs.items()
            if k.startswith("term_")
        },
        "n_trials": len(study.trials),
        "pinned_note": (
            "cut.axial_extent and cut.offset_max_frac are PINNED (not fitted): "
            "global stats cannot constrain focal-plane-cut geometry. gain/enf are "
            "degenerate -- randomize over them at generation."
        ),
    }
    fitted_json.write_text(json.dumps(payload, indent=2))

    # Full resolved config for downstream synthetic generation.
    fitted_cfg_path = out_dir / out_cfg.get("fitted_config", "fitted_config.yaml")
    with open(fitted_cfg_path, "w") as f:
        yaml.safe_dump(best_sim_cfg, f, sort_keys=False)

    # Comparison plots: side-by-side images + overlaid PSD + matched stats table.
    rng = np.random.default_rng(int(cal_cfg["optuna"].get("seed", 0)) + 7)
    sim_img = _sim_to_unit(simulate_image(best_sim_cfg, rng), best_sim_cfg)
    sim_stats = _simulated_stats(best_sim_cfg, stats_cfg, int(cal_cfg["optuna"].get("n_sim_images", 6)), rng)
    real_img = real_images[0]

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes[0, 0].imshow(real_img, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("real (561 nm lipid)")
    axes[0, 0].axis("off")
    axes[0, 1].imshow(sim_img, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("simulated (fitted)")
    axes[0, 1].axis("off")

    axes[1, 0].loglog(np.arange(1, len(real_stats["rapsd"])), real_stats["rapsd"][1:], label="real")
    axes[1, 0].loglog(np.arange(1, len(sim_stats["rapsd"])), sim_stats["rapsd"][1:], label="simulated")
    axes[1, 0].set_title("radial PSD")
    axes[1, 0].set_xlabel("spatial frequency (radius)")
    axes[1, 0].set_ylabel("power")
    axes[1, 0].legend()

    axes[1, 1].axis("off")
    rows = [
        ("mean", real_stats["mean"], sim_stats["mean"]),
        ("median (p50)", real_stats["p50"], sim_stats["p50"]),
        ("p99", real_stats["p99"], sim_stats["p99"]),
        ("p99.9", real_stats["p999"], sim_stats["p999"]),
        ("skewness", real_stats["skew"], sim_stats["skew"]),
    ]
    table = axes[1, 1].table(
        cellText=[[n, f"{r:.4f}", f"{s:.4f}"] for n, r, s in rows],
        colLabels=["statistic", "real", "simulated"],
        loc="center",
        cellLoc="center",
    )
    table.scale(1, 2)
    axes[1, 1].set_title("matched summary statistics")

    fig.suptitle(f"Calibration comparison (discrepancy = {study.best_value:.4g})", fontsize=14)
    fig.tight_layout()
    plot_path = out_dir / out_cfg.get("comparison_plot", "comparison.png")
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return fitted_json, fitted_cfg_path, plot_path


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main():
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(repo_root / "configs" / "calibration_smoke.yaml"))
    args = parser.parse_args()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("calibrate")

    with open(args.config) as f:
        cal_cfg = yaml.safe_load(f)

    base_cfg = load_config(repo_root / cal_cfg["base_sim_config"])
    stats_cfg = cal_cfg.get("statistics", {})

    log.info("Loading real images ...")
    real_images = load_real_images(cal_cfg["data"])
    real_stats = average_statistics([compute_statistics(im, stats_cfg) for im in real_images])
    log.info(f"  {len(real_images)} real images; real mean={real_stats['mean']:.4f} "
             f"p99={real_stats['p99']:.4f} p99.9={real_stats['p999']:.4f} skew={real_stats['skew']:.3f}")

    log.info(f"Running Optuna: {cal_cfg['optuna']['n_trials']} trials, "
             f"{cal_cfg['optuna'].get('n_sim_images', 6)} sim images/trial ...")
    study = run_calibration(cal_cfg, base_cfg, real_stats)
    log.info(f"Best discrepancy: {study.best_value:.6g}")
    for k, v in study.best_trial.params.items():
        log.info(f"  {k} = {v:.4g}")

    fitted_json, fitted_cfg, plot_path = _write_outputs(
        cal_cfg, base_cfg, study, real_stats, real_images, stats_cfg
    )
    log.info(f"Wrote {fitted_json}")
    log.info(f"Wrote {fitted_cfg}")
    log.info(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
