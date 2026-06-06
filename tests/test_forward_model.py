"""Tests for the GUV forward simulator."""

import copy
from pathlib import Path

import numpy as np
import pytest

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.forward_model import (
    GUV,
    _render_guv,
    load_config,
    simulate_batch,
    simulate_image,
)


def _disable_distractors(cfg):
    """Turn off the OOF haze and aggregate populations (for isolation tests)."""
    cfg["background"]["oof_count"] = 0
    cfg["background"]["oof_density"] = None
    cfg["aggregates"]["count"] = 0
    cfg["aggregates"]["density"] = None
    return cfg

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "sim_default.yaml"


@pytest.fixture
def config():
    return load_config(CONFIG_PATH)


def test_output_shape_and_dtype(config):
    rng = np.random.default_rng(0)
    img = simulate_image(config, rng)
    assert img.shape == (512, 512)
    assert img.dtype == np.float32


def test_returns_truth_list(config):
    rng = np.random.default_rng(1)
    img, truth = simulate_image(config, rng, return_truth=True)
    assert isinstance(truth, list)
    # Each entry is (x, y, apparent_diameter).
    for entry in truth:
        assert len(entry) == 3


def test_count_overrides_density(config):
    cfg = copy.deepcopy(config)
    cfg["guvs"]["count"] = 7
    rng = np.random.default_rng(2)
    _, truth = simulate_image(cfg, rng, return_truth=True)
    assert len(truth) == 7


def test_guv_count_scales_with_density(config):
    """Higher density -> more GUVs on average (averaged over many frames)."""
    low = copy.deepcopy(config)
    low["guvs"]["count"] = None
    low["guvs"]["density"] = 5
    high = copy.deepcopy(config)
    high["guvs"]["count"] = None
    high["guvs"]["density"] = 40

    rng = np.random.default_rng(3)
    n_frames = 30
    low_counts = [len(simulate_image(low, rng, return_truth=True)[1]) for _ in range(n_frames)]
    high_counts = [len(simulate_image(high, rng, return_truth=True)[1]) for _ in range(n_frames)]

    assert np.mean(high_counts) > np.mean(low_counts)


def test_ground_truth_apparent_diameters_in_range(config):
    cfg = copy.deepcopy(config)
    cfg["guvs"]["count"] = None
    cfg["guvs"]["density"] = 50
    d_min, d_max = cfg["guvs"]["d_min"], cfg["guvs"]["d_max"]

    rng = np.random.default_rng(4)
    for _ in range(10):
        _, truth = simulate_image(cfg, rng, return_truth=True)
        for _x, _y, apparent_diameter in truth:
            assert d_min <= apparent_diameter <= d_max


def test_ground_truth_excludes_out_of_focus_background(config):
    """Ground truth contains in-focus GUVs only -- never the OOF background.

    With a fixed in-focus count and a heavy out-of-focus population, the truth
    list length must equal the in-focus count exactly.
    """
    cfg = copy.deepcopy(config)
    cfg["guvs"]["count"] = 4
    cfg["background"]["oof_count"] = 25  # many haze blobs, none should be labeled
    cfg["background"]["oof_density"] = None

    rng = np.random.default_rng(7)
    _, truth = simulate_image(cfg, rng, return_truth=True)
    assert len(truth) == 4


def test_out_of_focus_background_adds_signal(config):
    """The OOF layer actually contributes diffuse signal to the image."""
    base = copy.deepcopy(config)
    base["guvs"]["count"] = 0  # isolate the background
    base["noise"] = {
        "optical_bg": 0.0,
        "shot_noise": False,
        "gain": 1.0,
        "enf": 1.0,
        "read_noise": 0.0,
        "offset": 0.0,
    }

    base["aggregates"]["count"] = 0  # isolate the OOF layer from aggregates
    base["aggregates"]["density"] = None

    with_bg = copy.deepcopy(base)
    with_bg["background"]["oof_count"] = 15
    with_bg["background"]["oof_density"] = None

    without_bg = copy.deepcopy(base)
    without_bg["background"]["oof_count"] = 0
    without_bg["background"]["oof_density"] = None

    img_bg = simulate_image(with_bg, np.random.default_rng(1))
    img_none = simulate_image(without_bg, np.random.default_rng(1))

    assert img_bg.sum() > img_none.sum()
    np.testing.assert_array_equal(img_none, np.zeros_like(img_none))


def test_aggregates_add_signal_but_excluded_from_ground_truth(config):
    """Saturated aggregates contribute signal yet never appear in ground truth."""
    base = copy.deepcopy(config)
    base["guvs"]["count"] = 3
    base["background"]["oof_count"] = 0  # isolate aggregates from the haze
    base["background"]["oof_density"] = None

    with_agg = copy.deepcopy(base)
    with_agg["aggregates"]["count"] = 6
    with_agg["aggregates"]["density"] = None

    without_agg = copy.deepcopy(base)
    without_agg["aggregates"]["count"] = 0
    without_agg["aggregates"]["density"] = None

    img_agg, truth_agg = simulate_image(with_agg, np.random.default_rng(2), return_truth=True)
    img_none, truth_none = simulate_image(without_agg, np.random.default_rng(2), return_truth=True)

    # Aggregates add bright signal...
    assert img_agg.sum() > img_none.sum()
    # ...but are NOT labeled: only the 3 in-focus GUVs are in the ground truth.
    assert len(truth_agg) == 3
    assert len(truth_none) == 3


def test_multiscale_oof_dots_add_high_frequency_structure(config):
    """Adding the fine-dot sub-population raises high-frequency haze power.

    Blobs render before dots, so with a fixed seed the blob layer is identical
    whether or not dots are present; the only difference is the dots. The
    multi-scale haze (blobs + dots) therefore has strictly more mid/high-frequency
    power than blobs alone -- that's the fine structure real haze has."""
    from src.statistics import radial_power_spectrum

    base = copy.deepcopy(config)
    base["guvs"]["count"] = 0
    base["aggregates"]["count"] = 0
    base["aggregates"]["density"] = None
    base["noise"] = {
        "optical_bg": 0.0, "shot_noise": False, "gain": 1.0,
        "enf": 1.0, "read_noise": 0.0, "offset": 0.0,
    }

    with_dots = copy.deepcopy(base)          # blobs + dots (default densities)
    blobs_only = copy.deepcopy(base)
    blobs_only["background"]["dot_density"] = 0  # same blobs, no fine dots

    ps_dots = radial_power_spectrum(simulate_image(with_dots, np.random.default_rng(0)))
    ps_blobs = radial_power_spectrum(simulate_image(blobs_only, np.random.default_rng(0)))

    hi = slice(40, 120)  # mid/high spatial-frequency band
    assert ps_dots[hi].mean() > ps_blobs[hi].mean()


def test_aggregates_saturate_at_sensor_max(config):
    """Bright aggregates clip to the configured sensor maximum."""
    cfg = copy.deepcopy(config)
    cfg["guvs"]["count"] = 0
    cfg["background"]["oof_count"] = 0
    cfg["background"]["oof_density"] = None
    cfg["aggregates"]["count"] = 8
    cfg["aggregates"]["density"] = None
    max_value = cfg["sensor"]["max_value"]

    img = simulate_image(cfg, np.random.default_rng(3))
    assert img.max() <= max_value
    # With amplitude well above sensor max, the brightest pixels should saturate.
    assert np.isclose(img.max(), max_value)


def _render_single(guv, cfg, seed=0):
    """Render one GUV on a clean canvas (no noise/bg) and return the image."""
    size = int(cfg["image"]["size"])
    img = np.zeros((size, size), dtype=np.float64)
    _render_guv(img, guv, cfg["ring"], cfg["cut"], np.random.default_rng(seed))
    return img


def test_equatorial_cut_is_ring_offequator_cut_is_disc(config):
    """Ring-vs-disc behavior emerges from cut geometry.

    A near-equatorial cut (offset ~ 0) renders a thin annulus with a DARK
    interior; an off-equator cut renders a filled disc with a BRIGHT interior.
    We assert on the center (interior) intensity relative to the membrane.
    """
    cfg = copy.deepcopy(config)
    cfg["ring"]["brightness_jitter"] = 0.0  # deterministic peak
    cfg["ring"]["rim_variation"] = 0.0      # uniform rim for a clean assertion
    cx = cy = 256.0

    # Near-equatorial cut: offset 0 -> thin ring.
    ring_guv = GUV(x=cx, y=cy, apparent_diameter=40.0, sphere_diameter=40.0, cut_offset=0.0)
    ring_img = _render_single(ring_guv, cfg)
    ring_center = ring_img[int(cy), int(cx)]
    ring_membrane = ring_img[int(cy), int(cx) + 20]  # at r ~ r_app = 20

    # Off-equator cut: chord radius 10 from a sphere cut well off-equator so the
    # fill smear (axial_extent*|h|/r_app) exceeds r_app -> filled disc.
    r_app = 10.0
    h = 30.0
    sphere_d = 2.0 * np.sqrt(r_app**2 + h**2)
    disc_guv = GUV(x=cx, y=cy, apparent_diameter=2 * r_app, sphere_diameter=sphere_d, cut_offset=h)
    disc_img = _render_single(disc_guv, cfg)
    disc_center = disc_img[int(cy), int(cx)]
    disc_membrane = disc_img[int(cy), int(cx) + int(r_app)]

    # Ring: dark center (hole), bright membrane.
    assert ring_center < 0.1 * ring_membrane
    # Disc: bright center, comparable to the membrane.
    assert disc_center > 0.5 * disc_membrane


def test_zeroed_noise_and_bg_gives_clean_image(config):
    """No-op noise + no background -> deterministic, noiseless ring image."""
    cfg = copy.deepcopy(config)
    cfg["guvs"]["count"] = 5
    cfg["ring"]["brightness_jitter"] = 0.0
    _disable_distractors(cfg)
    cfg["noise"] = {
        "optical_bg": 0.0,
        "shot_noise": False,
        "gain": 1.0,
        "enf": 1.0,
        "read_noise": 0.0,
        "offset": 0.0,
    }

    img_a = simulate_image(cfg, np.random.default_rng(123))
    img_b = simulate_image(cfg, np.random.default_rng(123))

    # Deterministic: same seed -> identical image (no stochastic noise).
    np.testing.assert_array_equal(img_a, img_b)
    # Noiseless signal is non-negative and contains actual ring brightness.
    assert img_a.min() >= 0.0
    assert img_a.max() > 0.0


def test_return_debug_exposes_cut_geometry(config):
    cfg = copy.deepcopy(config)
    cfg["guvs"]["count"] = 6
    rng = np.random.default_rng(9)
    img, truth, debug = simulate_image(cfg, rng, return_truth=True, return_debug=True)
    assert len(truth) == len(debug) == 6
    for g in debug:
        assert isinstance(g, GUV)
        # Apparent diameter never exceeds the true sphere diameter.
        assert g.apparent_diameter <= g.sphere_diameter + 1e-6


def test_simulate_batch_shape(config):
    cfg = copy.deepcopy(config)
    cfg["guvs"]["count"] = 3
    rng = np.random.default_rng(5)
    imgs, truths = simulate_batch(cfg, rng, n=4, return_truth=True)
    assert imgs.shape == (4, 512, 512)
    assert imgs.dtype == np.float32
    assert len(truths) == 4
    assert all(len(t) == 3 for t in truths)
