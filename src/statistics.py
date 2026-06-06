"""Summary statistics for a 512x512 lipid-channel image.

Used by the detection-free moment-matching calibration (see src/calibrate.py):
both real and simulated images are reduced to the same small set of summary
statistics, and Optuna minimizes a weighted discrepancy between them
(src/discrepancy.py).

All statistics are computed on a normalized image in [0, 1] (the caller divides
by the sensor full-scale, e.g. uint8 / 255, so real and simulated images share a
common intensity scale). The statistics are deliberately *global* (whole-image)
moments and spectra -- they capture brightness level, bright-tail behaviour, and
spatial texture, but NOT object-level geometry (which global stats cannot
constrain; see calibrate.py for the fit-vs-pinned split).

Statistics produced:
    mean         : mean pixel intensity (pulled up by the bright tail)
    p50          : median pixel intensity -- the HAZE-DOMINATED background level
                   (most pixels are background/haze, so this is distinct from the
                   mean and is what the haze-sensitive discrepancy term matches)
    p99, p999    : high quantiles (p99, p99.9) -- the bright tail (rings/junk)
    skew         : skewness of the pixel distribution (sparse-bright -> positive)
    hist         : normalized intensity histogram (for a Wasserstein distance)
    hist_centers : bin centers for `hist`
    rapsd        : radially-averaged power spectral density (1D) -- spatial texture
                   (a low-frequency band of this is matched explicitly to pin the
                   large-scale haze power; see src/discrepancy.py)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import skew

# Defaults; overridable via the stats config block.
_DEFAULT_HIST_BINS = 64


def radial_power_spectrum(img: np.ndarray) -> np.ndarray:
    """Radially-averaged power spectral density of a 2D image.

    The DC component is removed (mean-subtracted) so the curve describes texture,
    not overall brightness. Returns a 1D array indexed by integer spatial
    frequency (radius in the 2D power spectrum), length = min(H, W) // 2.
    """
    h, w = img.shape
    f = np.fft.fftshift(np.fft.fft2(img - img.mean()))
    power = (f.real**2 + f.imag**2) / (h * w)

    cy, cx = h // 2, w // 2
    yy, xx = np.indices((h, w))
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2).astype(int)

    radial_sum = np.bincount(r.ravel(), weights=power.ravel())
    radial_cnt = np.bincount(r.ravel())
    radial = radial_sum / np.maximum(radial_cnt, 1)
    return radial[: min(cy, cx)]


def compute_statistics(img: np.ndarray, stats_cfg: dict | None = None) -> dict:
    """Reduce one [0, 1] image to its summary statistics (see module docstring)."""
    stats_cfg = stats_cfg or {}
    n_bins = int(stats_cfg.get("hist_bins", _DEFAULT_HIST_BINS))

    flat = np.asarray(img, dtype=np.float64).ravel()
    hist, edges = np.histogram(flat, bins=n_bins, range=(0.0, 1.0), density=False)
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total  # normalize to a probability mass function
    centers = 0.5 * (edges[:-1] + edges[1:])

    return {
        "mean": float(flat.mean()),
        "p50": float(np.quantile(flat, 0.50)),
        "p99": float(np.quantile(flat, 0.99)),
        "p999": float(np.quantile(flat, 0.999)),
        "skew": float(skew(flat)),
        "hist": hist,
        "hist_centers": centers,
        "rapsd": radial_power_spectrum(np.asarray(img, dtype=np.float64)),
    }


def average_statistics(stats_list: list) -> dict:
    """Average a list of per-image statistics into one target/summary dict.

    Scalars and arrays are averaged elementwise (arrays must share length, which
    they do for fixed image size and bin count).
    """
    if not stats_list:
        raise ValueError("average_statistics: empty list")

    keys_scalar = ("mean", "p50", "p99", "p999", "skew")
    keys_array = ("hist", "rapsd")

    out: dict = {}
    for k in keys_scalar:
        out[k] = float(np.mean([s[k] for s in stats_list]))
    for k in keys_array:
        out[k] = np.mean(np.stack([s[k] for s in stats_list], axis=0), axis=0)
    # hist_centers is identical across images; carry the first.
    out["hist_centers"] = stats_list[0]["hist_centers"]
    return out
