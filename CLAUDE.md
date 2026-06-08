# CLAUDE.md — developer & physics notes for GUV-detect

Context for working in this repo: structure, conventions, the physics behind the
forward simulator, and which knobs are placeholders awaiting calibration.

## Project goal

End-to-end GUV detector: physics-based **forward simulator** → calibrate to real
images → generate synthetic images with ground-truth `(x, y, apparent_diameter)` →
train a CenterNet-style detector → test on real images. **Done so far:** the
scaffold, the forward simulator (rings + discs, small-heavy sizes, out-of-focus
haze, saturated aggregate distractors), the **detection-free moment-matching
calibration** that fits the simulator to the real 561 nm images, the
**labeled synthetic-dataset generator** (train/val, in-focus GUVs only), and the
**CenterNet-style detector** (small U-Net, heatmap + radius heads, train /
detect / evaluate). Next: run training on the H100 and test on real images.

## Repository structure

```
configs/sim_default.yaml      All simulator parameters. Single source of truth.
configs/calibration.yaml      Real calibration run (fit ranges, weights, I/O).
configs/calibration_smoke.yaml  Fast smoke calibration (few trials).
configs/dataset.yaml          Dataset generation (base config, ranges, split).
configs/train.yaml            Detector training hyperparameters.
configs/eval.yaml             Detector evaluation settings.
src/forward_model.py          simulate_image / simulate_batch + helpers.
src/statistics.py             Per-image summary statistics for calibration.
src/discrepancy.py            Config-driven weighted real-vs-sim discrepancy.
src/calibrate.py              Optuna calibration entry point + comparison plots.
src/generate_dataset.py       Labeled synthetic-dataset generator (train/val).
src/normalize.py              SHARED uint8 -> [0,1] normalization (train + detect).
src/dataset.py                torch Dataset + CenterNet target construction.
src/model.py                  Small U-Net, heatmap + radius heads (GUVNet).
src/train.py                  Training loop (focal + L1, AMP, warmup->cosine).
src/detect.py                 Inference: decode peaks + NMS -> (x, y, radius).
src/metrics.py                Greedy matching, precision/recall/F1, radius MAE.
src/evaluate.py               Val-set metrics overall + binned by crowding.
src/__init__.py
scripts/preview_synthetic.py  Renders preview grids (crowded fields, ring/disc
                              demo, size sweep) -> PNG.
scripts/preview_dataset.py    Overlays GT labels on generated images -> PNG.
scripts/detect_real.py        Runs detector on real images -> 3-panel PNGs.
scripts/sweep_thresholds.py   Runs detect_real.py at a list of thresholds.
run_all.sh                    generate (PNG) -> train -> evaluate (H100).
tests/test_forward_model.py   simulator pytest suite.
tests/test_calibration.py     statistics + discrepancy pytest suite.
tests/test_generate_dataset.py  dataset-generator pytest suite.
tests/test_detector.py        targets / decode / metrics / normalization suite.
data/                         Real .tif/.tiff stacks. DO NOT TOUCH.
models/                       Trained checkpoint(s); models/best.pt (gitignored,
                              dir kept via .gitkeep).
results/                      Inference experiment outputs, results/thresh_<value>/
                              per run (gitignored, dir kept via .gitkeep).
previews/                     Generated preview PNGs (gitignored output).
calibration_out[_smoke]/      Generated calibration outputs (gitignored).
dataset/                      Generated synthetic dataset (gitignored output).
runs/                         Training checkpoints + viz (gitignored output).
pyproject.toml                uv-managed project + deps.
```

## Conventions

- **Sizing is in DIAMETER (px)** everywhere a user sees it: config (`d_min`,
  `d_max`, ring `thickness`) and ground truth `(x, y, apparent_diameter)`.
  `d_min`/`d_max` bound the **apparent** (image / chord) diameter; the **true
  sphere** diameter is sampled separately and is typically larger. Radius
  (`diameter / 2`) is an internal detail.
- **No magic numbers / paths in `src/`.** Every physics parameter is read from
  the config. The one numeric constant in code is `_FWHM_TO_SIGMA` (a pure math
  definition for converting a width to a Gaussian sigma), not a tunable.
- **Coordinates:** `x` = column (axis 1), `y` = row (axis 0).
- **Randomness:** everything flows through a passed-in `np.random.Generator`
  (`np.random.default_rng(seed)`), so runs are reproducible.
- **Units:** ring/haze/aggregate amplitudes and `optical_bg` are in **photons**;
  gain converts to detector **counts**. Output image is `float32` counts clipped
  to `[0, sensor.max_value]` (default `[0, 255]`). Calibration normalizes by
  `sensor.max_value` to compare against the uint8 real images on a `[0, 1]` scale.
- Run things via `uv run ...`.

## Forward model — physics assumptions

The simulator (`src/forward_model.py`) renders one 512×512 single-channel
(lipid / 561 nm) image as follows.

1. **In-focus GUV population** (`_sample_guvs`)
   - Count: `Poisson(density)`, or exactly `count` if set (deterministic; used
     by tests/previews).
   - Center: **soft-exclusion (minimum-separation) placement** — overlap is
     **decoupled from density**. Real GUVs mostly exclude one another (pack/tile)
     rather than pile up, so each candidate center is rejected if it falls closer
     than `min_separation_factor * (r_i + r_j)` (apparent radii) to an
     already-placed GUV, via rejection sampling capped at `placement_max_attempts`
     (last candidate kept on exhaustion, so crowded frames stay finite). A
     minority — probability `allowed_overlap_fraction` per GUV — **skips the rule**
     so genuinely overlapping / nested cases are still represented. This lets
     density fill a frame by **packing** instead of by overlapping. Shapes near
     the border still clip off the image (realistic, intentional).
     `min_separation_factor: 0` recovers the old uniform placement. The knobs live
     in `guvs.*` (`configs/sim_default.yaml`); generation overrides them from
     `configs/dataset.yaml` `placement:`.
   - **Size distribution** (feature 2): the **true sphere diameter** is drawn
     **lognormal** (`size_dist: lognormal`, params `sphere_diameter_log_mean` /
     `_log_sigma`), which is small-dominated with a long thin tail — matching
     real GUV populations far better than a uniform draw. The focal-plane cut
     (below) then reduces it to the **apparent** diameter, which is
     **rejection-sampled** into `[d_min, d_max]` so labels stay in range while
     the underlying sizes/cuts stay physical. The distribution shape is a
     `[CALIBRATE]` placeholder to check against the real apparent-size histogram.
     `_sample_sphere_diameter` also supports `size_dist: mixture` — a
     two-component (small + large) lognormal mixture used by **dataset
     generation** to better cover both ends; see the dataset section. Calibration
     keeps the single `lognormal`.

2. **Focal-plane-cut model — rings AND filled discs** (feature 1;
   `_sample_one_guv`, `_render_guv`). *Assumption:* a GUV is a thin spherical
   shell of radius `R`; the confocal plane cuts it at height `h` above center
   (`|h| ≤ R`, default `h ~ U(−R, R)·offset_max_frac` — uniform over the
   sphere). The membrane's intersection is a circle of **chord radius**
   `r_app = √(R²−h²)`, so the **apparent diameter = 2·r_app ≤ sphere diameter**.
   Ring-vs-disc **emerges from geometry**: the confocal collects over a finite
   axial slab, so we add a radial fill **smear = `axial_extent`·|h|/r_app** (the
   radius-change-per-height at the cut). The in-focus membrane then fills the
   band `[r_app − smear, r_app]` with soft Gaussian edges of width `thickness`:
   - **equatorial cut** (`h ≈ 0`): `smear ≈ 0` → thin **ring**, dark interior;
   - **off-equator / polar cut**: `smear ≥ r_app` → band reaches the centre →
     bright **filled disc**.
   The **outer** radius stays at `r_app`, so the rendered extent always matches
   the apparent-diameter label. **Softness/variability** (real rings are soft,
   low-contrast and uneven, not crisp clean circles): `thickness` (edge width) is
   **fittable** — larger → softer, lower-contrast membrane; `brightness_jitter`
   varies peak brightness per GUV; and **`rim_variation`** modulates brightness
   *around the rim* (a sum of `rim_modes` random angular harmonics), so the ring
   breaks up / is uneven instead of a uniform circle. Rendered in a clipped
   bounding box for speed (this is what makes edge-clipping fall out naturally).

3. **In-focus PSF** — the clean in-focus layer is convolved with a **Gaussian
   PSF** (`psf.sigma`), the standard approximation of microscope blur.

4. **Structured, MULTI-SCALE out-of-focus background** (feature 3;
   `_render_oof_background`). A *separate* haze population rendered **behind** the
   in-focus layer. Real haze has structure at **all spatial scales**, so this is
   a **mixture of scales** (each object a Gaussian bump with its own blur, not a
   flat disc + one global blur):
   - **blobs** — per-object blur sigma drawn **log-uniform** over
     `blob_sigma_min..blob_sigma_max` (small-heavy: many small, few large);
   - **dots** — a *dense* sub-population of small, faint, near-in-focus dots
     (`dot_sigma_min..max`, `dot_density`) adding fine mid/high-frequency texture.
   A single global **`oof_amplitude`** scales both (the one calibration-fitted
   haze level); `blob_rel_amplitude`/`dot_rel_amplitude` set the mix; the
   per-scale **shape is pinned** (like the cut geometry). Master switch
   `oof_count`: `null` → use Poisson densities; `0` → disabled; `N>0` → exactly
   `N` objects split by the blob:dot density ratio. This is a dominant
   false-positive source, so it is deliberately **NOT** in the ground truth. The
   scalar `noise.optical_bg` remains as a flat floor on top of this.

5. **Saturated aggregate distractors** (feature 4; `_render_aggregates`). A
   *separate* population of bright, irregular, high-amplitude blobs modelling
   clumped lipid. Each is a few overlapping jittered Gaussian **lobes** (lumpy,
   not a clean circle); the amplitude is well above the ring brightness so it
   **clips flat at the sensor max** (saturates). Params under `aggregates`:
   `density`/`count`, `amplitude` (+`_jitter`), `lobe_sigma_min`/`_max`,
   `n_lobes`, `lobe_jitter`. Like the haze, these are distractors **NOT** in the
   ground truth — they teach the detector to ignore bright junk and judge by
   shape.

6. **PMT-style noise model** (`_apply_noise`), applied in this order — each
   stage is a separate, clearly-named config parameter:
   1. `optical_bg` — scalar background-**photon floor** (stray light / dark
      counts) added *before* shot noise, on top of the structured OOF layer.
   2. **shot noise** — `Poisson(signal + optical_bg)`. Toggle `shot_noise`
      (false → deterministic; used for noiseless reference renders).
   3. `gain` — counts per photo-electron (PMT amplification).
   4. `enf` — **excess noise factor** (≥1). The stochastic PMT multiplication
      inflates variance beyond pure shot noise; modeled as added Gaussian
      variance so total variance ≈ `gain²·enf²·N` (vs `gain²·N` for shot noise
      alone). `enf = 1` adds nothing.
   5. `read_noise` — Gaussian read noise (counts, std).
   6. `offset` — constant ADC pedestal.

7. **Sensor stage** (`_apply_sensor`) — clip to `[0, sensor.max_value]`
   (default 255). The **upper** clip is the saturation that flattens bright
   aggregates; the lower clip removes negative counts after read noise. Output
   is `float32` in `[0, max_value]`.

   **No-op (clean, noiseless ring) configuration** — also disable the OOF layer
   *and* aggregates (`background.oof_count: 0`, `aggregates.count: 0`); the test
   helper `_disable_distractors` does this:
   `optical_bg=0, shot_noise=false, gain=1, enf=1, read_noise=0, offset=0`.

### Placeholders awaiting calibration

Everything tagged `[CALIBRATE]` in `configs/sim_default.yaml`. By group:
- **size:** `d_min`, `d_max`, `sphere_diameter_log_mean`, `_log_sigma`;
- **cut:** `axial_extent` (`offset_max_frac` is a modelling choice, not tagged);
- **membrane:** ring `thickness` / `brightness` / `brightness_jitter` /
  `rim_variation`;
- **PSF:** `psf.sigma`;
- **out-of-focus:** `background.oof_amplitude` (global level); the multi-scale
  shape (`blob_*`, `dot_*`) is pinned;
- **aggregates:** `aggregates.density`, `aggregates.amplitude`;
- **noise:** all `noise.*`.

Current values are plausible starting points, **not** measured from data. The
calibration below fits the constrained subset to the real 561 nm images in
`data/` (structure first, then statistics).

## Calibration — detection-free moment matching

Fit the simulator to the real images by matching **global lipid-channel
statistics** (no detector, no per-object labels). Pipeline:

1. Reduce each real image (`data/`, 561 nm lipid channel, `[0,1]` after `/255`)
   and each simulated image (`/sensor.max_value`) to summary statistics
   (`src/statistics.py`): `mean`, **`p50` (median)**, `p99`, `p99.9`, `skewness`,
   an intensity `histogram`, and the radially-averaged power spectrum (`rapsd`).
2. Optuna minimizes a **config-driven weighted discrepancy**
   (`src/discrepancy.py`): mean/median/quantiles/skewness as **relative squared
   errors** (shared scale), histogram via **Wasserstein** distance, PSD via **MSE
   of log10**. Each term is `{enabled, weight}` in the calibration config.
   **Haze-sensitive terms** (something only the haze can satisfy, so the
   optimizer can't trade haze away to sharpen rings):
   - **`median`** — the haze-dominated background level (p50), distinct from the
     bright-tail-pulled `mean`;
   - **`psd_low_band`** — log10 MSE of the PSD over the lowest radial-frequency
     bins (`1..max_bin`), i.e. power at **large spatial scales** = the haze.
3. Outputs (`calibration_out/`): `fitted_params.json`, a full `fitted_config.yaml`
   for downstream generation, and `comparison.png` (real vs sim image, overlaid
   radial PSD, matched-stats table).

### Fit vs. pinned (important)

Global statistics constrain brightness, the bright tail, and texture — but
**cannot** constrain the focal-plane-cut geometry (the ring/disc ratio is a
minority effect on global moments). So:

- **Fitted** (declared in `configs/calibration.yaml` `fit:`): `psf.sigma`,
  `ring.brightness`, **`ring.thickness`** (ring softness),
  `noise.{gain,enf,read_noise,optical_bg,offset}`, `guvs.density`,
  `guvs.sphere_diameter_log_{mean,sigma}`, `background.oof_amplitude`
  (global haze level), `aggregates.amplitude`.
- **Pinned** (held at `sim_default.yaml` values): `cut.axial_extent`,
  `cut.offset_max_frac`, the multi-scale haze **shape** (`background.blob_*`,
  `dot_*`), `ring.rim_variation`, and everything else not listed in `fit:`.
- **Degeneracy:** `gain`/`enf` are degenerate (only their product is
  constrained). They're fitted to a point, but downstream **generation should
  randomize** over them within range, not trust the point estimate.

### Running it

```bash
uv run python src/calibrate.py --config configs/calibration_smoke.yaml  # fast smoke (~8 trials)
uv run python src/calibrate.py --config configs/calibration.yaml        # real run (~300 trials, minutes)
```

The smoke config exists only to verify the pipeline runs end to end; its fitted
values are not meaningful. Tune ranges/weights in `configs/calibration.yaml`.

## Synthetic dataset generation

`src/generate_dataset.py` turns the calibrated model into a labeled training set:

```bash
# Full regen, parallel across all CPU cores (default --n-workers = os.cpu_count()):
uv run python src/generate_dataset.py --config configs/dataset.yaml --n-workers 16
# Eyeball the size-mixture + density distribution on ~12 fresh images first:
uv run python scripts/preview_dataset.py --generate 12
uv run python scripts/preview_dataset.py    # or: overlay GT labels on an existing run
```

**Parallel generation.** Generation is embarrassingly parallel (each image is
independent pure-numpy CPU work). `--n-workers N` (default `os.cpu_count()`,
`1` = serial) runs a `multiprocessing.Pool` of **`spawn`** workers, each a fresh
interpreter pinned to **single-threaded** BLAS — a Pool *initializer* sets
`OMP_/MKL_/OPENBLAS_NUM_THREADS=1` and numpy/forward-model imports are **lazy**
(inside the per-image task) so those env vars take effect before numpy loads.
Without this, N workers × N BLAS threads oversubscribe the cores and it gets
*slower*. Seeding is **by image index** (`SeedSequence(seed).spawn(2)` →
per-image child seeds), never by worker, so parallel output is **byte-identical**
to serial — parallelism changes only speed and (manifest is re-sorted) not even
order. The split ratio and per-image content are preserved exactly.

Output layout under `output.dir` (default `dataset/`):
```
images/{train,val}/<id>.{npy|png}     # npy float32 counts, or png uint8
labels/{train,val}/<id>.{json|csv}    # in-focus GUVs as (x, y, diameter)
manifest.csv                          # id, split, image, label, n_guvs
dataset_meta.json                     # config, seed, randomized vs fixed params
```

**Labels = in-focus GUVs ONLY.** The out-of-focus haze and the saturated
aggregates are distractors and are never written to the labels — the detector
must learn to ignore them. (`simulate_image`'s ground truth already excludes them;
the generator never invents extra labels.)

**Randomize the under-constrained, fix the constrained.** Calibration constrains
the physics but not the object population or the degenerate gain/enf split, so
those are sampled **per image** (uniform over `[lo, hi]`) in
`configs/dataset.yaml` `randomize:`:
- `guvs.density` over `[10, 150]` (uniform). Real fields hold ~40–80 GUVs; the
  floor stays at **10** so the detector still sees sparse fields and won't
  over-predict density on near-empty real regions, and the ceiling is **150** to
  fill even the densest real fields. The soft-exclusion placement (below) means a
  high density now packs the frame rather than piling overlapping rings;
- `cut.axial_extent` over a wide range (the ring↔disc ratio was never
  constrained, so the detector should see the whole continuum);
- `gain`/`enf` **jointly** — keep the product near the fitted value (with a small
  jitter), vary the split via `enf` (`gain = product/enf`), so the detector sees
  the full plausible noise range.

**Sphere-diameter size = a two-component mixture (deliberate POPULATION change,
NOT calibrated physics).** Calibration used a single lognormal, which cannot at
once pile mass at the small end (real fields are small-dominated, which the
global-moment fit under-weights) *and* keep a realistic minority of large
vesicles reaching `d_max`. So generation replaces it with a mixture
(`configs/dataset.yaml` `size_distribution:`, `guvs.size_dist: mixture` in
`src/forward_model.py:_sample_sphere_diameter`): each GUV is independently the
**SMALL** component with probability `small_fraction`, else the **LARGE**
component, then drawn from that component's lognormal. Five fixed (un-randomized)
knobs, meant to be tuned **by eye** against `scripts/preview_dataset.py
--generate 12`: `small_fraction` (default `0.72`), `small_log_mean` (`3.40` →
~30 px median), `small_log_sigma` (`0.45`), `large_log_mean` (`4.60` → ~100 px
median), `large_log_sigma` (`0.35`, tail toward ~150 px). Values are SPHERE
diameter (px); the focal-plane cut then reduces each to an **apparent** diameter
rejection-sampled into `[d_min, d_max] = [12, 150]` (existing logic). The single
`lognormal` path is unchanged, so **calibration is untouched**. The two
components are deliberately **wide and overlapping** (`small_log_sigma 0.60`,
`large_log_mean 4.40` / `large_log_sigma 0.50`) so the combined spread is
**continuous** (smooth small-dominated falloff with real density at medium sizes)
rather than two separate humps.

**GUV placement = soft-exclusion (deliberate POPULATION change, NOT physics).**
Generation also overrides the placement knobs via `configs/dataset.yaml`
`placement:` (injected into `guvs.*` by `build_image_config`, recorded in
`dataset_meta.json`): `min_separation_factor` (default `0.9`),
`allowed_overlap_fraction` (default `0.12`), `placement_max_attempts` (`30`). See
the forward-model population note above for the mechanics. This is what lets the
raised density fill crowded frames by packing while keeping a realistic minority
of overlapping/nested GUVs.

Everything the calibration *did* constrain stays fixed at the base
(`fitted_config.yaml`) values: `psf.sigma`, `ring.brightness`/`thickness`, the
haze params, `noise.{offset,read_noise,optical_bg}`, aggregates, sensor. The
split between randomized and fixed is documented in `configs/dataset.yaml` and
echoed into `dataset_meta.json`.

## Detector — CenterNet-style (heatmap + radius)

A small U-Net predicts, per pixel, a **center heatmap** (probability of a GUV
center) and a **radius**. It learns to fire on in-focus rings/discs and ignore
the (unlabeled) haze and aggregates.

```bash
./run_all.sh                       # generate(PNG) -> train -> evaluate (H100)
./run_all.sh --smoke               # tiny end-to-end sanity run
uv run python src/train.py    --config configs/train.yaml [--smoke]
uv run python src/evaluate.py --config configs/eval.yaml
uv run python src/detect.py   --checkpoint models/best.pt --image <png|tif>
uv run python scripts/detect_real.py                 # single threshold (0.30)
uv run python scripts/sweep_thresholds.py            # several thresholds at once
```

### Real-image inference (local experiments)

- **Model location:** the trained checkpoint lives at **`models/best.pt`**. It is
  the default for `scripts/detect_real.py` and `scripts/sweep_thresholds.py`, so
  no `--checkpoint` is needed. `models/` is gitignored (the `.pt` is large) but
  kept via `models/.gitkeep`.
- **Output convention:** `scripts/detect_real.py` auto-names its output folder by
  threshold → **`results/thresh_<value>/`** (e.g. `results/thresh_0.10/`,
  `results/thresh_0.30/`), so runs at different thresholds never overwrite. An
  explicit `--out` overrides. `results/` is gitignored (kept via `.gitkeep`).
- **Single threshold:** `uv run python scripts/detect_real.py --threshold 0.15`
  → writes `results/thresh_0.15/` (3-panel PNG per image — raw | detections
  overlay | heatmap — plus `_grid.png`).
- **Sweep:** `uv run python scripts/sweep_thresholds.py --thresholds 0.10 0.15 0.20 0.25 0.30`
  runs `detect_real.py` once per threshold (each to its own
  `results/thresh_<value>/`). Defaults to that list, `models/best.pt`, and
  `data/`. It dispatches `detect_real.py` as a subprocess so the inference path
  is identical to a single run.

`scripts/detect_real.py` is the **qualitative real-image transfer check** (no GT):
per real image it loads via the shared normalization, runs the `detect.py`
pipeline, and writes the 3-panel PNG + `_grid.png`. Tune `--threshold`/`--nms-dist`
to inspect transfer without retraining. (On a non-CUDA box pass `--device cpu`.)

Key pieces:
- **Shared normalization** (`src/normalize.py`): the ONE place uint8 → `[0,1]`
  happens (`/255`). Both `src/dataset.py` (training) and `src/detect.py`
  (inference on synthetic **and** real images) import the same `load_normalized`,
  so preprocessing can't drift. The dataset is generated as uint8 PNG to match
  the real uint8 images through this path.
- **Targets** (`src/dataset.py:build_targets`): Gaussian heatmap (sigma scaled to
  object radius, `clip(radius*sigma_scale, sigma_min, sigma_max)`, max-combined),
  radius written at center pixels, and a `reg_mask` so radius L1 is supervised
  **only at true centers**. Built at **full resolution** (`out_stride: 1`) so
  close centers in crowded fields stay separable — do not downsample heavily.
- **Model** (`src/model.py:GUVNet`): small U-Net (`base`, `depth`, `out_stride`);
  two 1×1 heads — heatmap (sigmoid, clamped) + radius (raw, in input pixels). The
  heatmap-head bias is initialized to −4.6 (rare positives), standard for focal.
- **Loss** (`src/train.py`): penalty-reduced **focal** loss on the heatmap +
  **L1** on radius at GT centers; config'd weights. AdamW; LR = linear warmup
  (`warmup_frac` of steps) → **cosine** to ~0; AMP (autocast + GradScaler) on
  CUDA. Logs train/val loss each epoch, saves `best.pt` by val loss, and dumps
  decoded val predictions (`runs/<name>/viz/epoch_NNN.png`). `--smoke` runs a
  tiny subset for a couple of epochs to verify the loop.
- **Decode** (`src/detect.py:decode`): 3×3 max-pool peak-pick above `threshold`,
  read radius at each peak, greedy **NMS** by center distance → `(x, y, radius,
  score)`. Reused by the train-time viz and by `evaluate.py`.
- **Metrics** (`src/metrics.py`, `src/evaluate.py`): greedy match by center
  distance within `tol_frac * gt_radius`; precision / recall / F1 + radius MAE on
  matched pairs. `evaluate.py` prints overall metrics **and** precision/recall
  binned by crowding (GUVs/image) — the reference curve in place of a Hough
  baseline.

`out_stride` must be the same in `configs/train.yaml` (`model.out_stride`) and
`configs/eval.yaml` (`detect.down_ratio`); default 1 (full res).

## Public API

- `load_config(path) -> dict`
- `simulate_image(config, rng, return_truth=False, return_debug=False)`
  → `image` | `(image, truth)` | `(image, truth, debug)`, where `image` is
  `float32 (512,512)`, `truth` is a list of `(x, y, apparent_diameter)` for the
  **in-focus GUVs only**, and `debug` is a list of `GUV` dataclasses (adds
  `sphere_diameter`, `cut_offset` for inspection).
- `simulate_batch(config, rng, n, return_truth=False) -> images | (images, truths)`
  where `images` is `float32 (n,512,512)`.
- `GUV` dataclass: `x, y, apparent_diameter, sphere_diameter, cut_offset`.

## Testing

`uv run pytest`. Simulator (`test_forward_model.py`): output shape/dtype; count
scales with density; apparent diameters within `[d_min, d_max]`; ground truth
excludes the out-of-focus background **and the aggregates** (both add signal);
aggregates saturate at the sensor max; the focal-plane cut produces a ring
(equatorial, dark interior) vs a filled disc (off-equator, bright interior);
`return_debug` exposes cut geometry; and the no-op noise + no-distractor config
yields a deterministic noiseless image. Calibration (`test_calibration.py`):
statistics keys/ranges, identical images → zero discrepancy, brighter images →
larger discrepancy, disabled terms skipped, averaging shapes. Dataset generator
(`test_generate_dataset.py`): distractors never labeled (zero in-focus GUVs +
heavy haze/aggregates → empty labels), label counts/values in range, files +
train/val split written, gain/enf product preserved while the split varies,
randomized params within their configured ranges. Detector (`test_detector.py`):
target shapes/values (heatmap peak = 1 at centers, radius/mask correct, off-frame
centers dropped); peak-pick recovers known centers and NMS suppresses duplicates;
metric matching on a toy case (TP/FP/FN, radius MAE); **train and detect share
the identical normalization function**; model forward shapes at stride 1 and 2.

## Gotchas

- `data/` is read-only input — never modify or write into it.
- **Normalization must stay shared:** training and inference both go through
  `src/normalize.load_normalized` (uint8 `/255`). Don't add a second, divergent
  normalization in either path — `test_detector.py` asserts they're the same fn.
- The dataset must be **PNG** for the detector (`configs/dataset.yaml`
  `output.image_format: png`) so synthetic matches the real uint8 images.
- Keep `model.out_stride` (train) == `detect.down_ratio` (eval), default 1.
- The preview script imports private helpers (`_render_clean`, `_apply_noise`,
  `_render_oof_background`) and the `GUV` dataclass to render the deterministic
  demo/sweep panels; keep their signatures stable or update the script.
- Ground truth holds **in-focus GUVs only** — the OOF haze **and the saturated
  aggregates** are intentionally unlabeled. Don't "fix" this by adding either to
  the truth list; teaching the detector to ignore them is the whole point.
- `d_min`/`d_max` bound the **apparent** diameter; the sampled **sphere**
  diameter is larger. Apparent diameter is rejection-sampled into range.
- Calibration compares on a `[0,1]` scale: real `/255`, sim `/sensor.max_value`.
  If you change `sensor.max_value`, keep `data.max_value` consistent with the
  real bit depth.
- `src/statistics.py` is a package module (`src.statistics`), not the stdlib
  `statistics` — it does not shadow it as long as you import `from src...`.
