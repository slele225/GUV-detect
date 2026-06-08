"""Physics-based forward simulator for GUVs in confocal fluorescence images.

A GUV (giant unilamellar vesicle) is a lipid sphere. Imaged at the confocal
focal plane, its membrane shows up as a bright shape whose form depends on WHERE
the focal plane cuts the sphere (see "Focal-plane-cut model" below). This module
renders synthetic 512x512 single-channel (lipid / 561 nm) images together with
exact ground truth (x, y, apparent_diameter) for every in-focus GUV, so a
detector can be trained on data with perfect labels.

The simulator targets *structural* realism against real 561 nm images, which
contain three things a plain ring renderer lacks and which this module models:

    (1) Rings AND filled discs, emerging from focal-plane-cut geometry.
    (2) A small-dominated (lognormal) size distribution, not uniform.
    (3) A structured, MULTI-SCALE out-of-focus background (soft blobs from GUVs
        above/below the focal plane, with per-object blur scales plus a dense
        sub-population of small faint dots, so the haze has structure at all
        spatial frequencies) -- a dominant false-positive source, which the
        detector must learn to ignore (so it is NOT in the ground truth).
    (4) Saturated aggregate distractors (bright, irregular clumped lipid that
        clips at the sensor max) -- also unlabeled, to teach the detector to
        ignore bright junk and judge by shape.

Pipeline (all parameters live in configs/sim_default.yaml; see that file and
CLAUDE.md for the physics rationale and which values await calibration):

    1. Sample N in-focus GUVs (Poisson density or fixed count). For each:
       - center: uniform over the frame (edge-clipping and overlap allowed)
       - true sphere diameter: lognormal (small-heavy), see (2)
       - cut offset h: where the focal plane slices the sphere, see (1)
       - apparent diameter = 2 * chord radius at that cut (what we label)
    2. Render each GUV as a ring/disc per the cut geometry, in photons.
    3. Convolve the in-focus layer with a Gaussian PSF (microscope blur).
    4. Render and add the out-of-focus background layer (heavily blurred) and the
       saturated aggregate distractor layer.
    5. Apply a PMT-style noise model: optical-background floor, Poisson shot
       noise, gain, excess-noise factor, Gaussian read noise, ADC offset.
    6. Apply the sensor stage: clip to [0, sensor max] (saturation).

FOCAL-PLANE-CUT MODEL (feature 1)
    A GUV is a thin spherical shell of radius R. The confocal focal plane cuts
    it at height h above the sphere center (|h| <= R). The membrane's
    intersection with the plane is a circle of chord radius
        r_app = sqrt(R^2 - h^2),
    so the apparent (image) diameter is 2*r_app -- always <= the true sphere
    diameter. Whether it reads as a ring or a filled disc emerges from geometry:
    the confocal collects light over a finite axial slab, and near the *pole* of
    the sphere (|h| -> R) the membrane is nearly tangent to that slab, so a wide
    radial band of membrane is in focus at once and the centre fills in. Near the
    *equator* (h ~ 0) the membrane crosses the slab steeply, giving a thin ring.
    We capture this with a radial "fill" smear
        smear = axial_extent * |h| / r_app
    (the radius change per unit height at the cut, times the slab extent). The
    in-focus membrane then occupies the annular band [r_app - smear, r_app]:
    equatorial cut -> smear ~ 0 -> thin ring; polar cut -> smear >= r_app ->
    the band reaches the centre -> bright filled disc. The OUTER radius stays at
    r_app, so the apparent-diameter label always matches the rendered extent.

CONVENTIONS
    - Sizing the user sees (config d_min/d_max, ground truth) is DIAMETER (px).
      `d_min`/`d_max` bound the APPARENT diameter. Radius = diameter / 2 is
      internal. The true sphere diameter is sampled, not bounded directly.
    - Coordinates: x is the column (axis 1), y is the row (axis 0).
    - All physics constants/parameters come from the config; none are hardcoded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.ndimage import gaussian_filter

# FWHM (full width at half maximum) of a Gaussian = sigma * this factor.
# Used only to convert a user-facing "thickness" (a width) into a Gaussian
# sigma. This is a definition, not a tunable.
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


@dataclass
class GUV:
    """One in-focus GUV.

    `x`, `y`, `apparent_diameter` are what the detector predicts (and what
    ground truth reports). `sphere_diameter` and `cut_offset` are the underlying
    focal-plane-cut geometry, kept for rendering and debugging.
    """

    x: float
    y: float
    apparent_diameter: float
    sphere_diameter: float
    cut_offset: float


def load_config(path: str | Path) -> dict:
    """Load a simulator YAML config into a plain nested dict."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _sample_count(pop_cfg: dict, rng: np.random.Generator) -> int:
    """Number of objects for one frame.

    If `count` is set in the config, use it exactly (deterministic, handy for
    tests and previews). Otherwise draw Poisson(density) so frame-to-frame
    counts vary naturally.
    """
    if pop_cfg.get("count") is not None:
        return int(pop_cfg["count"])
    return int(rng.poisson(pop_cfg["density"]))


def _sample_sphere_diameter(guv_cfg: dict, rng: np.random.Generator) -> float:
    """Draw one true sphere diameter (px) from the small-heavy size distribution.

    Two distributions are supported via `size_dist`:

    - "lognormal" (default; used by calibration): a single lognormal,
      diameter = exp(Normal(sphere_diameter_log_mean, sphere_diameter_log_sigma)).
      Naturally small-dominated with a long thin tail toward large vesicles. The
      shape parameters are CALIBRATE placeholders fit against real images.

    - "mixture" (used by dataset generation): a TWO-COMPONENT lognormal mixture.
      Each GUV is independently assigned to the SMALL component with probability
      `small_fraction`, else to the LARGE component; the chosen component's
      lognormal (its own `*_log_mean` / `*_log_sigma`) is then drawn. This is a
      deliberate POPULATION choice (not calibrated physics): a single lognormal
      cannot simultaneously pile mass at the small end AND keep a realistic
      minority of large vesicles. See configs/dataset.yaml for the five knobs.
    """
    dist = guv_cfg.get("size_dist", "lognormal")
    if dist == "lognormal":
        log_mean = float(guv_cfg["sphere_diameter_log_mean"])
        log_sigma = float(guv_cfg["sphere_diameter_log_sigma"])
    elif dist == "mixture":
        if rng.random() < float(guv_cfg["small_fraction"]):
            log_mean = float(guv_cfg["small_log_mean"])
            log_sigma = float(guv_cfg["small_log_sigma"])
        else:
            log_mean = float(guv_cfg["large_log_mean"])
            log_sigma = float(guv_cfg["large_log_sigma"])
    else:
        raise ValueError(f"unsupported size_dist: {dist!r}")
    return float(np.exp(rng.normal(log_mean, log_sigma)))


def _sample_one_guv(guv_cfg: dict, cut_cfg: dict, size: int, rng: np.random.Generator) -> GUV:
    """Sample a single in-focus GUV (geometry + position).

    Sphere diameter is small-heavy (lognormal). The cut offset h is the focal
    plane's height relative to the sphere center; by default it is uniform over
    the sphere (h ~ U(-R, R) scaled by `offset_max_frac`), so most cuts are
    off-equator -- exactly why real fields show a mix of rings and discs. The
    apparent diameter (2 * chord radius) is rejection-sampled into [d_min, d_max]
    so labels stay in range while the underlying sizes/cuts stay physical.
    """
    d_min = float(guv_cfg["d_min"])
    d_max = float(guv_cfg["d_max"])
    offset_max_frac = float(cut_cfg["offset_max_frac"])
    max_attempts = int(guv_cfg.get("sample_max_attempts", 100))

    sphere_d = d_min
    cut_offset = 0.0
    apparent_d = d_min
    for _ in range(max_attempts):
        sphere_d = _sample_sphere_diameter(guv_cfg, rng)
        radius = sphere_d / 2.0
        cut_offset = rng.uniform(-1.0, 1.0) * offset_max_frac * radius
        chord = radius**2 - cut_offset**2
        apparent_d = 2.0 * np.sqrt(chord) if chord > 0.0 else 0.0
        if d_min <= apparent_d <= d_max:
            break
    else:
        # Fallback (rare with sensible config): clamp to range as an equatorial
        # cut so the label still matches what we render.
        apparent_d = float(np.clip(apparent_d, d_min, d_max))
        sphere_d = max(sphere_d, apparent_d)
        cut_offset = 0.0

    x = rng.uniform(0.0, size)
    y = rng.uniform(0.0, size)
    return GUV(x=x, y=y, apparent_diameter=apparent_d, sphere_diameter=sphere_d, cut_offset=cut_offset)


def _too_close(guv: GUV, placed: list, min_sep_factor: float) -> bool:
    """True if `guv`'s center is closer than `min_sep_factor * (r_i + r_j)` to any
    already-placed GUV `j` (radii are APPARENT radii = the rendered extent). With
    `min_sep_factor = 1` two GUVs must be at least tangent (no overlap); < 1
    permits some overlap; > 1 forces a gap between them."""
    r_i = guv.apparent_diameter / 2.0
    for other in placed:
        req = min_sep_factor * (r_i + other.apparent_diameter / 2.0)
        dx = guv.x - other.x
        dy = guv.y - other.y
        if dx * dx + dy * dy < req * req:
            return True
    return False


def _sample_guvs(guv_cfg: dict, cut_cfg: dict, size: int, rng: np.random.Generator) -> list:
    """Sample the in-focus GUV population for one frame.

    Placement uses a SOFT-EXCLUSION (minimum-separation) model so that overlap is
    decoupled from density: real GUVs are physical objects that mostly exclude one
    another (they pack/tile) rather than pile up, so raising the density fills the
    frame by PACKING instead of by overlapping. For each GUV we reject candidate
    centers that fall closer than `min_separation_factor * (r_i + r_j)` to any
    already-placed GUV (apparent radii), via rejection sampling capped at
    `placement_max_attempts` attempts. With probability `allowed_overlap_fraction`
    a GUV skips the check entirely and is placed wherever it first landed, so a
    realistic MINORITY of genuinely overlapping / nested cases is kept. Setting
    `min_separation_factor = 0` disables exclusion (the old uniform placement)."""
    n = _sample_count(guv_cfg, rng)
    min_sep_factor = float(guv_cfg.get("min_separation_factor", 0.0))
    allowed_overlap = float(guv_cfg.get("allowed_overlap_fraction", 1.0))
    place_attempts = int(guv_cfg.get("placement_max_attempts", 30))

    placed: list = []
    for _ in range(n):
        guv = _sample_one_guv(guv_cfg, cut_cfg, size, rng)
        # Exclusion applies unless disabled, this is the first GUV, or this GUV is
        # in the allowed-overlap minority (drawn BEFORE the placement loop so the
        # RNG stream is independent of how many attempts placement happens to use).
        enforce = min_sep_factor > 0.0 and placed and rng.random() >= allowed_overlap
        if enforce:
            for _attempt in range(place_attempts):
                if not _too_close(guv, placed, min_sep_factor):
                    break
                guv.x = rng.uniform(0.0, size)
                guv.y = rng.uniform(0.0, size)
            # On exhausting attempts we keep the last candidate -- a capped retry
            # means very crowded frames stay realistic instead of looping forever.
        placed.append(guv)
    return placed


def _render_guv(
    image: np.ndarray,
    guv: GUV,
    ring_cfg: dict,
    cut_cfg: dict,
    rng: np.random.Generator,
) -> None:
    """Add one GUV's ring/disc (in photons) into `image` in place.

    The in-focus membrane occupies the annular band [r_app - smear, r_app] (see
    the focal-plane-cut model in the module docstring), with soft Gaussian edges
    of width set by the config `thickness`:

        - r in [inner, r_app]      -> peak           (in-focus membrane band)
        - r >  r_app               -> peak * Gaussian falloff (outer edge)
        - r <  inner               -> peak * Gaussian falloff (inner edge)

    where inner = max(r_app - smear, 0) and smear = axial_extent * |h| / r_app.
    Equatorial cut (h~0): inner ~ r_app -> thin ring. Polar cut: inner -> 0 ->
    bright filled disc. The outer radius stays at r_app so the rendered extent
    matches the apparent-diameter label. Width is set BEFORE the PSF blur.

    Softness/variability (real rings are soft, low-contrast and uneven, not crisp
    clean circles):
        - `thickness` (the edge width) is calibration-fittable -- larger gives a
          softer, lower-contrast membrane.
        - `brightness_jitter` varies the peak brightness per GUV.
        - `rim_variation` modulates brightness around the rim (a sum of random
          angular harmonics), breaking up the ring so it is not a uniform circle.

    Rendering is restricted to a bounding box around the apparent radius for
    speed; the box is clipped to the image, so edge GUVs draw partially.
    """
    r_app = guv.apparent_diameter / 2.0
    sigma_edge = max(ring_cfg["thickness"] * _FWHM_TO_SIGMA, 1e-6)

    # Radial fill smear from the finite axial collection slab (cut geometry).
    h = abs(guv.cut_offset)
    if r_app > 1e-6:
        smear = float(cut_cfg["axial_extent"]) * h / r_app
    else:
        smear = float(cut_cfg["axial_extent"]) * h
    inner = max(r_app - smear, 0.0)

    # Per-GUV brightness with optional lognormal jitter (mean ~ brightness).
    jitter = ring_cfg.get("brightness_jitter", 0.0)
    if jitter > 0.0:
        # Lognormal multiplier; -0.5*sigma^2 keeps the multiplier's mean ~1.
        mult = np.exp(rng.normal(-0.5 * jitter**2, jitter))
    else:
        mult = 1.0
    peak = ring_cfg["brightness"] * mult

    size = image.shape[0]
    # Bounding box: the lit extent reaches r_app plus a few edge sigmas outward.
    reach = r_app + 4.0 * sigma_edge
    x0 = max(int(np.floor(guv.x - reach)), 0)
    x1 = min(int(np.ceil(guv.x + reach)) + 1, size)
    y0 = max(int(np.floor(guv.y - reach)), 0)
    y1 = min(int(np.ceil(guv.y + reach)) + 1, size)
    if x0 >= x1 or y0 >= y1:
        return  # entirely off-frame

    yy, xx = np.mgrid[y0:y1, x0:x1]
    r = np.sqrt((xx - guv.x) ** 2 + (yy - guv.y) ** 2)

    outer_edge = peak * np.exp(-0.5 * ((r - r_app) / sigma_edge) ** 2)
    inner_edge = peak * np.exp(-0.5 * ((r - inner) / sigma_edge) ** 2)
    val = np.where(
        r > r_app,
        outer_edge,
        np.where(r >= inner, peak, inner_edge),
    )

    # Around-the-rim intensity variation: real membranes are uneven / broken,
    # not uniform clean circles. Modulate brightness by a sum of a few random
    # angular harmonics; where the modulation dips toward 0 the rim breaks up.
    rim_var = float(ring_cfg.get("rim_variation", 0.0))
    if rim_var > 0.0:
        n_modes = max(int(ring_cfg.get("rim_modes", 3)), 1)
        theta = np.arctan2(yy - guv.y, xx - guv.x)
        mod = np.ones_like(r)
        for _ in range(n_modes):
            m = int(rng.integers(1, 6))               # angular harmonic
            phase = rng.uniform(0.0, 2.0 * np.pi)
            amp_k = rng.uniform(0.3, 1.0)
            mod += (rim_var / n_modes) * amp_k * np.cos(m * theta + phase)
        val = val * np.clip(mod, 0.0, None)

    image[y0:y1, x0:x1] += val


def _render_clean(guvs: list, size: int, ring_cfg: dict, cut_cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """Sum all in-focus GUV rings/discs into a clean photon image (no PSF/noise)."""
    image = np.zeros((size, size), dtype=np.float64)
    for guv in guvs:
        _render_guv(image, guv, ring_cfg, cut_cfg, rng)
    return image


def _render_oof_population(
    layer: np.ndarray,
    size: int,
    n: int,
    base_amp: float,
    amp_jitter: float,
    sigma_min: float,
    sigma_max: float,
    sigma_log: bool,
    rng: np.random.Generator,
) -> None:
    """Add `n` soft Gaussian blobs (one OOF sub-population) into `layer` in place.

    Each blob carries its OWN blur scale: sigma is sampled per object from
    [sigma_min, sigma_max] -- log-uniform when `sigma_log` (a small-heavy mix:
    many small, few large). Modelling each out-of-focus object directly as a
    Gaussian bump (instead of a flat disc + one global blur) is what gives the
    haze structure across spatial frequencies.
    """
    if n <= 0 or base_amp <= 0.0:
        return
    log_min, log_max = np.log(max(sigma_min, 1e-3)), np.log(max(sigma_max, 1e-3))
    for _ in range(n):
        x = rng.uniform(0.0, size)
        y = rng.uniform(0.0, size)
        if sigma_log:
            sigma = float(np.exp(rng.uniform(log_min, log_max)))
        else:
            sigma = float(rng.uniform(sigma_min, sigma_max))
        amp = base_amp
        if amp_jitter > 0.0:
            amp = base_amp * np.exp(rng.normal(-0.5 * amp_jitter**2, amp_jitter))

        reach = int(np.ceil(4.0 * sigma)) + 1
        x0 = max(int(np.floor(x)) - reach, 0)
        x1 = min(int(np.ceil(x)) + reach, size)
        y0 = max(int(np.floor(y)) - reach, 0)
        y1 = min(int(np.ceil(y)) + reach, size)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        rr2 = (xx - x) ** 2 + (yy - y) ** 2
        layer[y0:y1, x0:x1] += amp * np.exp(-0.5 * rr2 / sigma**2)


def _render_oof_background(size: int, bg_cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """Render the structured, MULTI-SCALE out-of-focus background (in photons).

    Models GUVs above/below the focal plane as soft, low-contrast blobs rendered
    BEHIND the in-focus GUVs. Real haze has structure at ALL spatial scales, so
    this is a MIXTURE of scales rather than a single blur:

      - blobs : per-object blur sigma drawn (log-uniform) across a wide range ->
                many small, faintly-blurred objects plus a few large, heavily-
                blurred ones.
      - dots  : a denser sub-population of small, faint, near-in-focus dots that
                add fine mid-/high-frequency structure.

    A single global `oof_amplitude` scales BOTH populations (it is the one
    calibration-fitted haze level); per-population `*_rel_amplitude` set the mix.
    The per-scale SHAPE (densities, sigma ranges) is pinned, like the cut
    geometry -- global stats constrain the overall level, not the fine shape.
    These are background ONLY -- never returned as ground truth.

    Master switch `oof_count`: None -> use per-population Poisson densities;
    0 -> disabled (zero layer); >0 -> render exactly that many objects total,
    split between the two populations by their density ratio.
    """
    layer = np.zeros((size, size), dtype=np.float64)
    if bg_cfg is None:
        return layer

    blob_density = float(bg_cfg.get("blob_density", 0.0))
    dot_density = float(bg_cfg.get("dot_density", 0.0))

    oof_count = bg_cfg.get("oof_count")
    if oof_count is not None:
        if int(oof_count) <= 0:
            return layer  # master disable
        total_density = blob_density + dot_density
        if total_density <= 0:
            return layer
        n_blob = int(round(int(oof_count) * blob_density / total_density))
        n_dot = int(oof_count) - n_blob
    else:
        n_blob = int(rng.poisson(blob_density))
        n_dot = int(rng.poisson(dot_density))

    global_amp = float(bg_cfg.get("oof_amplitude", 0.0))
    amp_jitter = float(bg_cfg.get("oof_amplitude_jitter", 0.0))

    # (a) Multi-scale blurred blobs (small-heavy via log-uniform sigma).
    _render_oof_population(
        layer, size, n_blob,
        global_amp * float(bg_cfg.get("blob_rel_amplitude", 1.0)), amp_jitter,
        float(bg_cfg.get("blob_sigma_min", 2.0)), float(bg_cfg.get("blob_sigma_max", 40.0)),
        bool(bg_cfg.get("blob_sigma_log", True)), rng,
    )
    # (b) Fine structure: small faint near-in-focus dots.
    _render_oof_population(
        layer, size, n_dot,
        global_amp * float(bg_cfg.get("dot_rel_amplitude", 0.5)), amp_jitter,
        float(bg_cfg.get("dot_sigma_min", 0.8)), float(bg_cfg.get("dot_sigma_max", 2.5)),
        bool(bg_cfg.get("dot_sigma_log", False)), rng,
    )
    return layer


def _render_aggregates(size: int, agg_cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """Render the saturated aggregate distractor layer (in photons).

    Models clumped lipid: bright, irregular, high-amplitude blobs that saturate
    the sensor (they clip at the sensor max in `_apply_sensor`). Each aggregate
    is a few overlapping, jittered Gaussian lobes -- lumpy rather than a clean
    circle -- at an amplitude well above the ring brightness so it clips flat.

    Like the out-of-focus haze, these are distractors ONLY: they are deliberately
    NOT returned as ground truth, so the detector must learn to ignore bright
    junk and judge GUVs by shape. Returns a zero layer if disabled.
    """
    layer = np.zeros((size, size), dtype=np.float64)
    if agg_cfg is None:
        return layer
    n = _sample_count({"density": agg_cfg.get("density", 0.0), "count": agg_cfg.get("count")}, rng)
    if n <= 0:
        return layer

    amplitude = float(agg_cfg["amplitude"])
    amp_jitter = float(agg_cfg.get("amplitude_jitter", 0.0))
    sigma_min = float(agg_cfg["lobe_sigma_min"])
    sigma_max = float(agg_cfg["lobe_sigma_max"])
    n_lobes = int(agg_cfg.get("n_lobes", 3))
    lobe_jitter = float(agg_cfg.get("lobe_jitter", 6.0))

    for _ in range(n):
        cx = rng.uniform(0.0, size)
        cy = rng.uniform(0.0, size)
        amp = amplitude
        if amp_jitter > 0.0:
            amp = amplitude * np.exp(rng.normal(-0.5 * amp_jitter**2, amp_jitter))

        for _lobe in range(max(n_lobes, 1)):
            lx = cx + rng.normal(0.0, lobe_jitter)
            ly = cy + rng.normal(0.0, lobe_jitter)
            sigma = rng.uniform(sigma_min, sigma_max)
            lobe_amp = amp * rng.uniform(0.5, 1.0)  # uneven lobe brightness -> lumpy

            reach = int(np.ceil(4.0 * sigma)) + 1
            x0 = max(int(np.floor(lx)) - reach, 0)
            x1 = min(int(np.ceil(lx)) + reach, size)
            y0 = max(int(np.floor(ly)) - reach, 0)
            y1 = min(int(np.ceil(ly)) + reach, size)
            if x0 >= x1 or y0 >= y1:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1]
            rr2 = (xx - lx) ** 2 + (yy - ly) ** 2
            layer[y0:y1, x0:x1] += lobe_amp * np.exp(-0.5 * rr2 / sigma**2)

    return layer


def _apply_sensor(counts: np.ndarray, sensor_cfg: dict) -> np.ndarray:
    """Sensor digitization stage: clip to the valid range (saturation).

    Clips at `max_value` (the sensor full-scale, e.g. 255 for an 8-bit image),
    which is what makes the bright aggregates saturate to a flat top. The lower
    clip at 0 prevents negative counts after read noise. If `max_value` is unset,
    only the lower clip is applied.
    """
    max_value = sensor_cfg.get("max_value") if sensor_cfg else None
    if max_value is None:
        return np.clip(counts, 0.0, None)
    return np.clip(counts, 0.0, float(max_value))


def _apply_noise(photons: np.ndarray, noise_cfg: dict, rng: np.random.Generator) -> np.ndarray:
    """Apply the PMT-style detector noise model to a photon image.

    Stages, in order (each governed by a separate config parameter):
        1. optical_bg : scalar background-photon FLOOR added before shot noise
                        (stray light / dark counts), on top of the structured
                        out-of-focus background already in `photons`.
        2. shot noise : Poisson on (signal + optical_bg). Toggle with
                        `shot_noise` (false -> deterministic, for noiseless
                        reference renders).
        3. gain       : counts per photo-electron (PMT amplification).
        4. enf        : excess noise factor (>=1). The stochastic PMT gain
                        inflates variance beyond pure shot noise; we model this
                        as extra Gaussian variance so total variance becomes
                        gain^2 * enf^2 * N (vs gain^2 * N for shot noise alone).
                        enf = 1 adds nothing.
        5. read_noise : Gaussian read noise (counts, std).
        6. offset     : constant ADC pedestal.

    No-op configuration (clean, noiseless image):
        optical_bg=0, shot_noise=false, gain=1, enf=1, read_noise=0, offset=0
    """
    optical_bg = noise_cfg["optical_bg"]
    gain = noise_cfg["gain"]
    enf = noise_cfg["enf"]
    read_noise = noise_cfg["read_noise"]
    offset = noise_cfg["offset"]
    shot = noise_cfg.get("shot_noise", True)

    # 1. Background-photon floor (added before shot noise).
    photons = photons + optical_bg

    # 2. Photon shot noise.
    if shot:
        photons = rng.poisson(np.clip(photons, 0.0, None)).astype(np.float64)

    # 3. Gain (photons -> counts).
    counts = photons * gain

    # 4. Excess-noise factor: extra Gaussian variance from the PMT gain process.
    if enf > 1.0:
        excess_var = (gain**2) * (enf**2 - 1.0) * np.clip(photons, 0.0, None)
        counts = counts + rng.normal(0.0, np.sqrt(excess_var))

    # 5. Gaussian read noise.
    if read_noise > 0.0:
        counts = counts + rng.normal(0.0, read_noise, size=counts.shape)

    # 6. Constant ADC offset.
    counts = counts + offset

    return counts


def simulate_image(
    config: dict,
    rng: np.random.Generator,
    return_truth: bool = False,
    return_debug: bool = False,
):
    """Render one 512x512 single-channel (lipid) image of GUVs.

    Args:
        config: nested dict (see configs/sim_default.yaml).
        rng: numpy random Generator (np.random.default_rng(...)) for
            reproducibility; all randomness flows through it.
        return_truth: if True, also return the ground-truth list (in-focus GUVs
            only) as (x, y, apparent_diameter) tuples.
        return_debug: if True, also return the full list of GUV records
            (including sphere_diameter and cut_offset) for inspection. Implies
            return_truth in the returned tuple.

    Returns:
        image                              (default)
        image, truth                       (return_truth)
        image, truth, debug                (return_debug)
    where image is float32 (size, size), truth is a list of
    (x, y, apparent_diameter), and debug is a list of GUV dataclasses.
    """
    size = int(config["image"]["size"])

    # In-focus GUV layer: sample, render rings/discs, blur with the PSF.
    guvs = _sample_guvs(config["guvs"], config["cut"], size, rng)
    clean = _render_clean(guvs, size, config["ring"], config["cut"], rng)
    in_focus = gaussian_filter(clean, sigma=config["psf"]["sigma"], mode="constant")

    # Out-of-focus background layer (heavily blurred), added behind the GUVs.
    oof = _render_oof_background(size, config["background"], rng)

    # Saturated aggregate distractors (clumped lipid) -- bright, unlabeled junk.
    aggregates = _render_aggregates(size, config.get("aggregates"), rng)

    photons = in_focus + oof + aggregates
    noisy = _apply_noise(photons, config["noise"], rng)
    image = _apply_sensor(noisy, config.get("sensor")).astype(np.float32)
    truth = [(g.x, g.y, g.apparent_diameter) for g in guvs]

    if return_debug:
        return image, truth, guvs
    if return_truth:
        return image, truth
    return image


def simulate_batch(
    config: dict,
    rng: np.random.Generator,
    n: int,
    return_truth: bool = False,
):
    """Render `n` images.

    Returns:
        images float32 (n, size, size). If return_truth, also a list of length
        n, each entry the ground-truth GUV list (in-focus only) for that image.
    """
    images = []
    truths = []
    for _ in range(n):
        out = simulate_image(config, rng, return_truth=return_truth)
        if return_truth:
            img, truth = out
            images.append(img)
            truths.append(truth)
        else:
            images.append(out)

    images = np.stack(images, axis=0)
    if return_truth:
        return images, truths
    return images
