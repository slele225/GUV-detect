"""Tests for the GUV detector: targets, decoding, metrics, shared normalization."""

from pathlib import Path

import numpy as np
import pytest
import torch

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import src.dataset as dataset_mod
import src.detect as detect_mod
from src.dataset import build_targets
from src.detect import decode, nms_by_distance
from src.metrics import aggregate, match_detections, precision_recall_f1
from src.model import GUVNet
from src.normalize import load_normalized, to_unit


# --------------------------------------------------------------------------- #
# (1) Target construction
# --------------------------------------------------------------------------- #
def test_build_targets_shapes_and_values():
    size = 128
    guvs = [(40.0, 50.0, 20.0), (100.0, 90.0, 30.0)]  # (x, y, diameter)
    hm, radius_map, mask = build_targets(guvs, size, down_ratio=1, sigma_scale=0.15)

    assert hm.shape == (size, size) == radius_map.shape == mask.shape
    assert hm.dtype == np.float32
    # Heatmap peaks at exactly 1.0 at each center; radius written there.
    for x, y, d in guvs:
        cx, cy = int(round(x)), int(round(y))
        assert hm[cy, cx] == pytest.approx(1.0)
        assert radius_map[cy, cx] == pytest.approx(d / 2.0)
        assert mask[cy, cx] == 1.0
    assert mask.sum() == len(guvs)
    assert 0.0 <= hm.min() and hm.max() <= 1.0


def test_build_targets_downsample_and_offframe():
    size = 64
    # One in-frame GUV and one whose center is off-frame (should be dropped).
    guvs = [(30.0, 30.0, 16.0), (1000.0, 5.0, 10.0)]
    hm, radius_map, mask = build_targets(guvs, size, down_ratio=2)
    assert hm.shape == (32, 32)
    assert mask.sum() == 1  # off-frame center not labeled


# --------------------------------------------------------------------------- #
# (2) Peak-picking recovers known centers
# --------------------------------------------------------------------------- #
def test_decode_recovers_known_centers():
    size = 128
    guvs = [(20.0, 30.0, 16.0), (90.0, 100.0, 24.0), (60.0, 40.0, 12.0)]
    hm, radius_map, _ = build_targets(guvs, size, down_ratio=1)

    dets = decode(torch.from_numpy(hm), torch.from_numpy(radius_map),
                  threshold=0.5, nms_dist=5.0, down_ratio=1)

    assert len(dets) == len(guvs)
    # Each known center is recovered by some detection (within 1 px), radius too.
    for x, y, d in guvs:
        dist = np.hypot(dets[:, 0] - x, dets[:, 1] - y)
        j = int(np.argmin(dist))
        assert dist[j] <= 1.0
        assert dets[j, 2] == pytest.approx(d / 2.0, abs=0.5)


def test_nms_suppresses_close_duplicates():
    # Two near-identical detections + one far: NMS keeps the stronger + the far one.
    dets = np.array([
        [50.0, 50.0, 10.0, 0.9],
        [52.0, 51.0, 10.0, 0.7],   # within nms_dist of the first -> dropped
        [100.0, 100.0, 8.0, 0.6],
    ])
    kept = nms_by_distance(dets, min_dist=6.0)
    assert len(kept) == 2
    assert [50.0, 50.0] in kept[:, :2].tolist()
    assert [100.0, 100.0] in kept[:, :2].tolist()


# --------------------------------------------------------------------------- #
# (3) Metric matching on a toy case
# --------------------------------------------------------------------------- #
def test_metric_matching_toy_case():
    # GT: two GUVs. Preds: one close hit, one slightly-off hit, one false positive.
    gts = [(50.0, 50.0, 20.0), (100.0, 100.0, 20.0)]  # radius 10 each, tol=0.5*10=5
    preds = np.array([
        [51.0, 50.0, 9.0, 0.9],    # matches GT0 (dist 1 <= 5); radius err 1
        [100.0, 103.0, 11.0, 0.8],  # matches GT1 (dist 3 <= 5); radius err 1
        [10.0, 10.0, 5.0, 0.7],     # no GT nearby -> false positive
    ])
    res = match_detections(preds, gts, tol_frac=0.5)
    assert res["tp"] == 2
    assert res["fp"] == 1
    assert res["fn"] == 0

    p, r, f1 = precision_recall_f1(res["tp"], res["fp"], res["fn"])
    assert p == pytest.approx(2 / 3)
    assert r == pytest.approx(1.0)

    agg = aggregate([res])
    assert agg["radius_mae"] == pytest.approx(1.0)


def test_metric_misses_count_as_fn():
    gts = [(50.0, 50.0, 20.0), (200.0, 200.0, 20.0)]
    preds = np.array([[50.0, 50.0, 10.0, 0.9]])  # only finds one
    res = match_detections(preds, gts, tol_frac=0.5)
    assert res["tp"] == 1 and res["fp"] == 0 and res["fn"] == 1


# --------------------------------------------------------------------------- #
# (4) Train and detect use the SAME normalization
# --------------------------------------------------------------------------- #
def test_train_and_detect_share_normalization(tmp_path):
    from PIL import Image

    arr = (np.arange(256, dtype=np.uint8).reshape(16, 16))
    png = tmp_path / "img.png"
    Image.fromarray(arr).save(png)

    # The dataset module and the detect module must reference the identical fn.
    assert dataset_mod.load_normalized is detect_mod.load_normalized

    loaded = load_normalized(png)
    assert loaded.dtype == np.float32
    assert loaded.max() <= 1.0
    np.testing.assert_allclose(loaded, arr.astype(np.float32) / 255.0)
    np.testing.assert_allclose(to_unit(arr), arr.astype(np.float32) / 255.0)


# --------------------------------------------------------------------------- #
# (5) Model forward shapes
# --------------------------------------------------------------------------- #
def test_model_forward_shapes_full_res():
    model = GUVNet(in_ch=1, base=8, depth=3, out_stride=1).eval()
    with torch.no_grad():
        out = model(torch.zeros(2, 1, 64, 64))
    assert out["hm"].shape == (2, 1, 64, 64)
    assert out["radius"].shape == (2, 1, 64, 64)
    assert float(out["hm"].min()) >= 0.0 and float(out["hm"].max()) <= 1.0


def test_model_forward_shapes_strided():
    model = GUVNet(in_ch=1, base=8, depth=3, out_stride=2).eval()
    with torch.no_grad():
        out = model(torch.zeros(1, 1, 64, 64))
    assert out["hm"].shape == (1, 1, 32, 32)
