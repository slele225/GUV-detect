"""Tests for the synthetic dataset generator.

The central guarantee: only IN-FOCUS GUVs are ever labeled -- out-of-focus haze
and saturated aggregates are distractors and must never appear in the labels.
"""

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.forward_model import load_config
from src.generate_dataset import build_image_config, generate_dataset

REPO = Path(__file__).resolve().parents[1]
SIM_CONFIG = REPO / "configs" / "sim_default.yaml"


@pytest.fixture
def base_cfg():
    # Use the default sim config as the calibrated base so tests don't depend on
    # calibration_out/ existing.
    return load_config(SIM_CONFIG)


def _size_cfg():
    return {
        "size_dist": "mixture",
        "small_fraction": 0.72,
        "small_log_mean": 3.40,
        "small_log_sigma": 0.45,
        "large_log_mean": 4.60,
        "large_log_sigma": 0.35,
    }


def _place_cfg():
    return {
        "min_separation_factor": 0.9,
        "allowed_overlap_fraction": 0.12,
        "placement_max_attempts": 30,
    }


def _ds_cfg(n_images=6, val_ratio=0.0, density=(10.0, 150.0)):
    return {
        "base_config": "configs/sim_default.yaml",
        "output": {"dir": "unused", "image_format": "npy", "label_format": "json"},
        "dataset": {"n_images": n_images, "val_ratio": val_ratio, "seed": 0},
        "randomize": {
            "guv_density": list(density),
            "cut_axial_extent": [1.0, 8.0],
            "enf": [1.0, 1.6],
            "product_jitter": [0.85, 1.20],
        },
        "size_distribution": _size_cfg(),
        "placement": _place_cfg(),
    }


def _read_labels(out_dir, label_rel):
    data = json.loads((Path(out_dir) / label_rel).read_text())
    return data["guvs"]


def _manifest_rows(out_dir):
    with open(Path(out_dir) / "manifest.csv", newline="") as f:
        return list(csv.DictReader(f))


def test_distractors_are_never_labeled(tmp_path, base_cfg):
    """Zero in-focus GUVs + heavy haze and aggregates -> every label file empty.

    This directly proves the distractor populations never leak into the labels.
    """
    base = copy.deepcopy(base_cfg)
    base["background"]["oof_count"] = 60   # heavy haze
    base["aggregates"]["count"] = 20       # many bright aggregates
    base["aggregates"]["density"] = None

    ds_cfg = _ds_cfg(n_images=5, density=(0.0, 0.0))  # no in-focus GUVs at all
    generate_dataset(ds_cfg, base, tmp_path)

    rows = _manifest_rows(tmp_path)
    assert len(rows) == 5
    for r in rows:
        assert int(r["n_guvs"]) == 0
        assert _read_labels(tmp_path, r["label"]) == []


def test_label_counts_and_values_reasonable(tmp_path, base_cfg):
    """With normal density, labels exist, are in-range, and within the frame."""
    base = copy.deepcopy(base_cfg)
    size = base["image"]["size"]
    d_min, d_max = base["guvs"]["d_min"], base["guvs"]["d_max"]

    ds_cfg = _ds_cfg(n_images=8, density=(15.0, 30.0))
    summary = generate_dataset(ds_cfg, base, tmp_path)

    assert summary["total_labels"] > 0  # some GUVs were labeled
    for r in _manifest_rows(tmp_path):
        labels = _read_labels(tmp_path, r["label"])
        assert int(r["n_guvs"]) == len(labels)
        assert len(labels) < 200  # sane upper bound
        for g in labels:
            assert d_min <= g["diameter"] <= d_max
            assert 0.0 <= g["x"] <= size
            assert 0.0 <= g["y"] <= size


def test_files_and_split_written(tmp_path, base_cfg):
    """Images, labels, manifest and meta are written with the configured split."""
    ds_cfg = _ds_cfg(n_images=10, val_ratio=0.2)
    summary = generate_dataset(ds_cfg, copy.deepcopy(base_cfg), tmp_path)

    assert summary["n_train"] == 8 and summary["n_val"] == 2
    assert (tmp_path / "manifest.csv").exists()
    assert (tmp_path / "dataset_meta.json").exists()

    rows = _manifest_rows(tmp_path)
    assert sum(r["split"] == "val" for r in rows) == 2
    for r in rows:
        assert (tmp_path / r["image"]).exists()
        assert (tmp_path / r["label"]).exists()
        img = np.load(tmp_path / r["image"])
        assert img.shape == (512, 512)
        assert img.dtype == np.float32


def test_gain_enf_product_preserved_split_varies(base_cfg):
    """Per-image gain/enf keep the product near fitted while the split varies."""
    base = copy.deepcopy(base_cfg)
    base_product = base["noise"]["gain"] * base["noise"]["enf"]
    rand = _ds_cfg()["randomize"]

    size = _size_cfg()
    enfs, products = [], []
    for i in range(40):
        rng = np.random.default_rng(i)
        cfg = build_image_config(base, rand, size, rng)
        enfs.append(cfg["noise"]["enf"])
        products.append(cfg["noise"]["gain"] * cfg["noise"]["enf"])

    # Product stays within the configured jitter band of the fitted product...
    assert all(0.80 * base_product <= p <= 1.25 * base_product for p in products)
    # ...while the enf split genuinely varies across its range.
    assert max(enfs) - min(enfs) > 0.3


def test_randomized_params_within_ranges(base_cfg):
    rand = _ds_cfg()["randomize"]
    size = _size_cfg()
    place = _place_cfg()
    for i in range(50):
        rng = np.random.default_rng(100 + i)
        cfg = build_image_config(copy.deepcopy(base_cfg), rand, size, rng, place_cfg=place)
        assert rand["guv_density"][0] <= cfg["guvs"]["density"] <= rand["guv_density"][1]
        assert rand["cut_axial_extent"][0] <= cfg["cut"]["axial_extent"] <= rand["cut_axial_extent"][1]
        assert cfg["guvs"]["count"] is None
        # The mixture knobs are copied into the simulator config unchanged.
        assert cfg["guvs"]["size_dist"] == "mixture"
        assert cfg["guvs"]["small_fraction"] == size["small_fraction"]
        # The soft-exclusion placement knobs are injected into the guv config.
        assert cfg["guvs"]["min_separation_factor"] == place["min_separation_factor"]
        assert cfg["guvs"]["allowed_overlap_fraction"] == place["allowed_overlap_fraction"]
