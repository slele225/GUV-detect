"""Detection metrics: greedy center-distance matching, precision/recall/F1, MAE.

A prediction matches a ground-truth GUV if their centers are within a tolerance
that scales with the GT radius (`tol_frac * gt_radius`). Matching is greedy by
prediction score (highest first), each GT used at most once. From the matches we
report precision, recall, F1, and the radius mean-absolute-error on matched
pairs.
"""

from __future__ import annotations

import numpy as np


def match_detections(preds: np.ndarray, gts: list, tol_frac: float = 0.5) -> dict:
    """Match predictions to ground truth.

    Args:
        preds: (N, 4) array of (x, y, radius, score) -- or (N,>=2); score optional.
        gts: list of (x, y, diameter) ground-truth GUVs.
        tol_frac: match radius = tol_frac * gt_radius (in pixels).

    Returns dict with tp, fp, fn, and `radius_abs_errors` (list, matched pairs).
    """
    preds = np.asarray(preds, dtype=np.float64)
    if preds.size == 0:
        preds = np.zeros((0, 4))
    gt = np.array([[x, y, d / 2.0] for (x, y, d) in gts], dtype=np.float64) if gts else np.zeros((0, 3))

    n_gt = len(gt)
    # Order predictions by score (col 3) if present, else as given.
    if len(preds) and preds.shape[1] >= 4:
        order = np.argsort(-preds[:, 3])
    else:
        order = np.arange(len(preds))

    matched_gt = np.zeros(n_gt, dtype=bool)
    tp = 0
    radius_errors = []
    for i in order:
        px, py, pr = preds[i, 0], preds[i, 1], preds[i, 2]
        best_j, best_d = -1, np.inf
        for j in range(n_gt):
            if matched_gt[j]:
                continue
            gx, gy, gr = gt[j]
            tol = tol_frac * gr
            d = np.hypot(px - gx, py - gy)
            if d <= tol and d < best_d:
                best_d, best_j = d, j
        if best_j >= 0:
            matched_gt[best_j] = True
            tp += 1
            radius_errors.append(abs(pr - gt[best_j, 2]))

    fp = len(preds) - tp
    fn = n_gt - tp
    return {"tp": tp, "fp": fp, "fn": fn, "radius_abs_errors": radius_errors}


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def aggregate(results: list) -> dict:
    """Combine per-image match dicts into overall metrics."""
    tp = sum(r["tp"] for r in results)
    fp = sum(r["fp"] for r in results)
    fn = sum(r["fn"] for r in results)
    errs = [e for r in results for e in r["radius_abs_errors"]]
    precision, recall, f1 = precision_recall_f1(tp, fp, fn)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "radius_mae": float(np.mean(errs)) if errs else float("nan"),
        "tp": tp, "fp": fp, "fn": fn, "n_matched": len(errs),
    }
