"""Config-driven weighted discrepancy between real and simulated image stats.

Given two summary-statistics dicts (from src/statistics.py), compute a single
scalar discrepancy that Optuna minimizes during calibration. Every term is
individually toggleable and weighted via the config:

    terms:
      mean:                  {enabled: true, weight: 1.0}
      median:                {enabled: true, weight: 3.0}
      p99:                   {enabled: true, weight: 0.3}
      p999:                  {enabled: true, weight: 0.3}
      skewness:              {enabled: true, weight: 0.5}
      histogram_wasserstein: {enabled: true, weight: 100.0}
      radial_psd:            {enabled: true, weight: 1.0}
      psd_low_band:          {enabled: true, weight: 3.0, max_bin: 8}

Scale convention:
    - mean / median / p99 / p999 / skewness are RELATIVE squared errors,
      ((sim - real) / (|real| + eps))^2, so they are dimensionless and a single
      shared weight keeps them on the same scale.
    - histogram_wasserstein is the 1D Wasserstein (earth-mover) distance between
      the two intensity histograms on [0, 1]; it is small in absolute terms, so
      its weight is typically larger to bring it onto the same scale.
    - radial_psd is the mean squared error of log10(PSD) across ALL spatial
      frequencies (PSD spans orders of magnitude, so we compare in log space).

Haze-sensitive terms (something only the haze can satisfy, so the optimizer
can't trade haze away to sharpen rings):
    - median is the relative squared error of the median (p50) -- the
      haze-dominated background level, distinct from the bright-tail-pulled mean.
    - psd_low_band is the log10 MSE of the PSD restricted to the LOWEST radial-
      frequency bins (1 .. max_bin, skipping DC), i.e. power at LARGE spatial
      scales -- the diffuse haze. `max_bin` is read from the term spec.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import wasserstein_distance

_EPS = 1e-8


def _rel_sq_err(sim: float, real: float) -> float:
    """Relative squared error, robust to real ~ 0."""
    return float(((sim - real) / (abs(real) + _EPS)) ** 2)


def _term_value(name: str, real: dict, sim: dict, spec: dict) -> float:
    if name == "mean":
        return _rel_sq_err(sim["mean"], real["mean"])
    if name == "median":
        return _rel_sq_err(sim["p50"], real["p50"])
    if name == "p99":
        return _rel_sq_err(sim["p99"], real["p99"])
    if name == "p999":
        return _rel_sq_err(sim["p999"], real["p999"])
    if name == "skewness":
        return _rel_sq_err(sim["skew"], real["skew"])
    if name == "histogram_wasserstein":
        return float(
            wasserstein_distance(real["hist_centers"], sim["hist_centers"], real["hist"], sim["hist"])
        )
    if name == "radial_psd":
        log_real = np.log10(np.asarray(real["rapsd"]) + _EPS)
        log_sim = np.log10(np.asarray(sim["rapsd"]) + _EPS)
        return float(np.mean((log_sim - log_real) ** 2))
    if name == "psd_low_band":
        # Power at LARGE spatial scales: lowest radial-frequency bins, skip DC.
        max_bin = int(spec.get("max_bin", 8))
        hi = max(max_bin, 2)
        log_real = np.log10(np.asarray(real["rapsd"])[1:hi] + _EPS)
        log_sim = np.log10(np.asarray(sim["rapsd"])[1:hi] + _EPS)
        return float(np.mean((log_sim - log_real) ** 2))
    raise ValueError(f"unknown discrepancy term: {name!r}")


def discrepancy(real: dict, sim: dict, disc_cfg: dict) -> tuple:
    """Return (total, breakdown) where breakdown maps term -> weighted value.

    `disc_cfg` is the `discrepancy` block of the calibration config (a `terms`
    mapping of {enabled, weight}). Disabled terms are skipped.
    """
    terms = disc_cfg["terms"]
    total = 0.0
    breakdown: dict = {}
    for name, spec in terms.items():
        if not spec.get("enabled", True):
            continue
        weight = float(spec.get("weight", 1.0))
        contribution = weight * _term_value(name, real, sim, spec)
        breakdown[name] = contribution
        total += contribution
    return total, breakdown
