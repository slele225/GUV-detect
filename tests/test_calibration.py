"""Tests for the calibration building blocks (statistics + discrepancy)."""

from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.discrepancy import discrepancy
from src.statistics import average_statistics, compute_statistics


def test_compute_statistics_keys_and_ranges():
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 1, size=(512, 512))
    s = compute_statistics(img)
    for key in ("mean", "p50", "p99", "p999", "skew", "hist", "hist_centers", "rapsd"):
        assert key in s
    assert 0.0 <= s["mean"] <= 1.0
    assert s["p999"] >= s["p99"] >= s["p50"]
    # Histogram is a normalized pmf.
    assert np.isclose(s["hist"].sum(), 1.0)


def test_identical_images_have_zero_discrepancy():
    rng = np.random.default_rng(1)
    img = rng.uniform(0, 1, size=(512, 512))
    s = compute_statistics(img)
    disc_cfg = {
        "terms": {
            "mean": {"enabled": True, "weight": 1.0},
            "median": {"enabled": True, "weight": 1.0},
            "p99": {"enabled": True, "weight": 1.0},
            "p999": {"enabled": True, "weight": 1.0},
            "skewness": {"enabled": True, "weight": 1.0},
            "histogram_wasserstein": {"enabled": True, "weight": 50.0},
            "radial_psd": {"enabled": True, "weight": 1.0},
            "psd_low_band": {"enabled": True, "weight": 3.0, "max_bin": 8},
        }
    }
    total, breakdown = discrepancy(s, s, disc_cfg)
    assert total == 0.0
    assert all(v == 0.0 for v in breakdown.values())
    assert "median" in breakdown and "psd_low_band" in breakdown


def test_brighter_image_increases_discrepancy():
    """A clearly different image has a larger discrepancy than a near-identical one."""
    rng = np.random.default_rng(2)
    base = rng.uniform(0, 0.3, size=(512, 512))
    near = np.clip(base + rng.normal(0, 0.001, base.shape), 0, 1)
    far = np.clip(base + 0.4, 0, 1)  # much brighter

    s_base = compute_statistics(base)
    s_near = compute_statistics(near)
    s_far = compute_statistics(far)

    disc_cfg = {
        "terms": {
            "mean": {"enabled": True, "weight": 1.0},
            "histogram_wasserstein": {"enabled": True, "weight": 50.0},
        }
    }
    near_total, _ = discrepancy(s_base, s_near, disc_cfg)
    far_total, _ = discrepancy(s_base, s_far, disc_cfg)
    assert far_total > near_total


def test_haze_sensitive_terms_respond_to_large_scale_power():
    """psd_low_band (and median) pick up haze the bright-tail terms would miss."""
    rng = np.random.default_rng(10)
    # Haze-poor: only fine high-frequency noise, near-zero large-scale power.
    flat = np.clip(rng.normal(0.1, 0.01, (512, 512)), 0, 1)
    # Haze-rich: add a few large smooth blobs -> large-scale (low-frequency) power.
    yy, xx = np.mgrid[0:512, 0:512]
    hazy = flat.copy()
    for cx, cy in [(128, 128), (380, 300), (250, 420)]:
        hazy += 0.15 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 80.0**2))
    hazy = np.clip(hazy, 0, 1)

    s_flat = compute_statistics(flat)
    s_hazy = compute_statistics(hazy)
    cfg = {"terms": {"psd_low_band": {"enabled": True, "weight": 1.0, "max_bin": 8}}}
    same, _ = discrepancy(s_flat, s_flat, cfg)
    diff, _ = discrepancy(s_flat, s_hazy, cfg)
    assert same == 0.0
    assert diff > 0.0


def test_disabled_terms_are_skipped():
    rng = np.random.default_rng(3)
    a = compute_statistics(rng.uniform(0, 1, (512, 512)))
    b = compute_statistics(rng.uniform(0, 1, (512, 512)))
    disc_cfg = {
        "terms": {
            "mean": {"enabled": True, "weight": 1.0},
            "p99": {"enabled": False, "weight": 1.0},
        }
    }
    total, breakdown = discrepancy(a, b, disc_cfg)
    assert "p99" not in breakdown
    assert "mean" in breakdown


def test_average_statistics_shapes():
    rng = np.random.default_rng(4)
    stats = [compute_statistics(rng.uniform(0, 1, (512, 512))) for _ in range(3)]
    avg = average_statistics(stats)
    assert avg["rapsd"].shape == stats[0]["rapsd"].shape
    assert avg["hist"].shape == stats[0]["hist"].shape
    assert isinstance(avg["mean"], float)
