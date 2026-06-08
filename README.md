# GUV-detect

Detect **GUVs** (giant unilamellar vesicles — lipid spheres) in confocal
fluorescence microscopy images. Imaged at the focal plane, a GUV membrane
appears as a thin bright **annulus** (ring).

## Roadmap

1. **Forward simulator** ✅ — a physics-based renderer that produces synthetic
   lipid-channel images with exact ground truth `(x, y, apparent_diameter)` for
   every in-focus GUV. Built to match the *structure* of real 561 nm fields,
   modelling four things a plain ring renderer lacks:
   - **Rings AND filled discs**, emerging from **focal-plane-cut geometry** — a
     cut near the sphere's equator gives a thin ring; a cut nearer a pole gives a
     smaller, brighter filled disc.
   - **Soft, uneven rings** (fittable membrane width + around-the-rim intensity
     variation) — real rings are low-contrast and broken, not crisp circles.
   - A **small-dominated (lognormal) size distribution**, not uniform.
   - A **structured, multi-scale out-of-focus background** — soft blobs with a
     range of blur scales plus a dense sub-population of small faint dots, so the
     haze has structure at all spatial frequencies (a dominant false-positive
     source).
   - **Saturated aggregate distractors** (bright, irregular clumped lipid that
     clips at the sensor max). Both the haze and the aggregates are left
     **unlabeled** so the detector learns to ignore junk and judge by shape.
2. **Calibration** ✅ — detection-free **moment matching**: fit the simulator's
   constrained parameters to the real 561 nm images in `data/` by matching global
   lipid-channel statistics with Optuna (see below).
3. **Synthetic dataset** ✅ — generate labeled images at scale from the calibrated
   model, randomizing the under-constrained parameters per image (see below).
4. **Detector** ✅ — a CenterNet-style U-Net (center heatmap + radius head)
   trained on the synthetic data; detects in-focus GUVs and ignores haze /
   aggregates (see below).
5. **Evaluation** — test on real images (the synthetic val metrics are in place).

Steps 1–4 exist so far.

## Layout

```
configs/   sim_default.yaml — every simulator parameter (no magic numbers in src)
           calibration.yaml / calibration_smoke.yaml — calibration settings
           dataset.yaml — dataset generation (base config, randomization, split)
           train.yaml / eval.yaml — detector training / evaluation settings
src/       forward_model.py — the simulator (simulate_image / simulate_batch)
           statistics.py / discrepancy.py — calibration building blocks
           calibrate.py — Optuna calibration entry point
           generate_dataset.py — labeled synthetic-dataset generator
           normalize.py — shared uint8 -> [0,1] (training AND inference)
           dataset.py / model.py — torch Dataset + targets, U-Net (GUVNet)
           train.py / detect.py — training loop, inference (decode + NMS)
           metrics.py / evaluate.py — matching, P/R/F1, MAE, crowding curve
scripts/   preview_synthetic.py — render preview grids to a PNG
           preview_dataset.py — overlay GT labels on generated images
           detect_real.py — run the detector on real images (transfer check)
           sweep_thresholds.py — run detect_real.py at several thresholds
run_all.sh — generate (PNG) -> train -> evaluate (H100)
tests/     pytest suites for simulator, calibration, dataset, and detector
data/      real .tif/.tiff microscope images (DO NOT MODIFY)
models/    trained checkpoint(s): models/best.pt (gitignored; dir kept)
results/   real-image inference outputs: results/thresh_<value>/ (gitignored)
```

## Setup

Uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
```

On Linux (e.g. the H100), `uv sync` installs a CUDA-enabled PyTorch wheel; on
other platforms it installs the CPU build (fine for tests and the simulator).

## Usage

```bash
# Render preview grids (crowded fields, ring/disc demo, size sweep) to a PNG.
uv run python scripts/preview_synthetic.py

# Run tests.
uv run pytest
```

```python
import numpy as np
from src.forward_model import load_config, simulate_image

config = load_config("configs/sim_default.yaml")
rng = np.random.default_rng(0)
image, truth = simulate_image(config, rng, return_truth=True)
# image: float32 (512, 512)
# truth: list of (x, y, apparent_diameter) in px -- in-focus GUVs only
#        (the out-of-focus background haze is intentionally NOT labeled)

# Need the underlying cut geometry too? (sphere_diameter, cut_offset)
image, truth, debug = simulate_image(config, rng, return_truth=True, return_debug=True)
```

## Calibration

Fit the simulator to the real images by matching **global lipid-channel
statistics** (no detector, no labels): both real (`/255`) and simulated
(`/sensor.max_value`) images are reduced to `mean`, `median (p50)`, `p99`,
`p99.9`, `skewness`, an intensity histogram, and the radial power spectrum;
Optuna minimizes a config-driven weighted discrepancy between them. Two terms are
**haze-sensitive** — `median` (haze-dominated background level) and `psd_low_band`
(power at large spatial scales) — so the optimizer can't trade the haze away to
sharpen rings.

```bash
# Fast smoke test (~8 trials) — verifies the pipeline runs end to end.
uv run python src/calibrate.py --config configs/calibration_smoke.yaml

# Real fit (~300 trials, minutes). Outputs go to calibration_out/.
uv run python src/calibrate.py --config configs/calibration.yaml
```

Outputs: `fitted_params.json`, a full `fitted_config.yaml` for downstream
generation, and `comparison.png` (real vs simulated image, overlaid radial PSD,
matched-stats table).

**What's fitted vs. pinned.** Global statistics can't constrain the
focal-plane-cut geometry (the ring/disc ratio is a minority effect on global
moments) or the fine multi-scale-haze *shape*, so `cut.axial_extent`,
`cut.offset_max_frac`, the haze shape (`background.blob_*` / `dot_*`), and
`ring.rim_variation` are **pinned**. Optuna fits only what the statistics
constrain: `psf.sigma`, ring brightness, **ring thickness (softness)**, the noise
model, in-focus density, the lognormal size params, and the global haze /
aggregate amplitudes. `gain`/`enf` are degenerate (only their product is
constrained) — fit to a point here, but **randomize over them when generating**
the dataset. See [CLAUDE.md](CLAUDE.md) and `configs/calibration.yaml` for the
full list and rationale.

## Synthetic dataset

Generate a labeled training set from the calibrated model:

```bash
# Generates N images (configs/dataset.yaml) -> dataset/, parallel across CPU
# cores (--n-workers defaults to os.cpu_count(); use 1 to run serially).
uv run python src/generate_dataset.py --config configs/dataset.yaml --n-workers 16

# Eyeball the size-mixture + density distribution on ~12 fresh images BEFORE a
# full regen (generates them on the fly at the current config settings):
uv run python scripts/preview_dataset.py --generate 12

# Or overlay the ground-truth labels on an already-generated dataset.
uv run python scripts/preview_dataset.py
```

**Parallel generation.** Each image is independent pure-numpy CPU work, so
generation fans out over a `multiprocessing.Pool` of **`spawn`** workers, each
pinned to **single-threaded** BLAS (a Pool initializer sets
`OMP_/MKL_/OPENBLAS_NUM_THREADS=1` and numpy is imported lazily in the worker so
that takes effect first — otherwise the per-worker BLAS thread pools
oversubscribe the cores and it runs *slower*). Per-image seeding is **by image
index**, never by worker, so the output is **byte-identical** to a serial run —
only speed changes.

Output under `dataset/`: `images/{train,val}/<id>.npy` (or `.png`),
`labels/{train,val}/<id>.json` (or `.csv`) with each in-focus GUV as
`(x, y, apparent_diameter)`, a `manifest.csv`, and a `dataset_meta.json`.

**Only in-focus GUVs are labeled** — the out-of-focus haze and the saturated
aggregates are distractors and are never written to the labels, so the detector
learns to ignore them and judge by shape.

**Per-image randomization of the under-constrained parameters** (calibration
fixed the physics, not the population or the degenerate noise split): GUV density
over `[10, 150]` (sparse→very crowded; real fields hold ~40–80, floor kept at 10
so sparse fields are still seen, ceiling at 150 to fill the densest),
`cut.axial_extent` (the full ring↔disc continuum), and `gain`/`enf` jointly
(product held near the fitted value, split varied). Everything calibration *did*
constrain stays fixed at the `fitted_config.yaml` values. All ranges, and the
randomized-vs-fixed split, live in `configs/dataset.yaml`.

**Soft-exclusion GUV placement** (a *population* change, distinct from physics):
overlap is decoupled from density. Real GUVs mostly exclude one another
(pack/tile), so each GUV rejects candidate centers closer than
`min_separation_factor * (r_i + r_j)` to an already-placed GUV, with a minority
(`allowed_overlap_fraction`) allowed to overlap/nest anyway. This lets the raised
density fill crowded frames by packing instead of piling overlapping rings. Knobs
are in `configs/dataset.yaml` `placement:` (defaults `min_separation_factor: 0.9`,
`allowed_overlap_fraction: 0.12`); `min_separation_factor: 0` recovers uniform
placement.

**Sphere-diameter size is a two-component mixture** (a deliberate *population*
change, distinct from the calibrated physics): a single lognormal can't both pile
mass at the small end *and* keep a realistic minority of large vesicles, so each
GUV is drawn from a SMALL or LARGE lognormal component (`small_fraction` picks
which). Five fixed knobs in `configs/dataset.yaml` `size_distribution:`
(`small_fraction`, `small/large_log_mean`, `small/large_log_sigma`) are meant to
be tuned by eye via `preview_dataset.py --generate 12`. Calibration still uses
the single lognormal — only generation uses the mixture.

> **Note:** the dataset is generated as **uint8 PNG** (`output.image_format: png`)
> so it matches the real 512×512 uint8 images. Training and inference share one
> normalization (`src/normalize.py`, `/255`) so they can't drift.

## Detector

A small U-Net (CenterNet-style) predicts a per-pixel **center heatmap** and a
**radius**. It learns to fire on in-focus rings/discs and ignore the unlabeled
haze and aggregates. Targets and the heatmap are at **full resolution** so close
centers in crowded fields stay separable.

```bash
# Full pipeline on the H100: generate(PNG) -> train -> evaluate.
DEVICE=cuda N_WORKERS=16 ./run_all.sh
./run_all.sh --smoke                 # tiny end-to-end sanity run

# Or step by step:
uv run python src/train.py    --config configs/train.yaml          # train
uv run python src/train.py    --config configs/train.yaml --smoke  # 50 imgs, 2 epochs
uv run python src/evaluate.py --config configs/eval.yaml           # val metrics
uv run python src/detect.py   --checkpoint runs/guvnet/best.pt --image <png|tif>
```

- **Loss:** penalty-reduced focal on the heatmap + L1 on radius at GT centers.
  AdamW; LR = linear warmup → cosine to ~0; mixed precision on CUDA. Each epoch
  logs train/val loss, saves `best.pt`, and writes decoded val predictions to
  `runs/guvnet/viz/epoch_NNN.png` so you can watch it learn.
- **Inference** (`detect.py`): peak-pick heatmap maxima above a threshold → read
  radius at each peak → NMS by center distance → `(x, y, radius)`. Runs on
  synthetic and real uint8 images **identically** via the shared normalization.
- **Evaluation** (`evaluate.py`): greedy center-distance matching within
  `tol_frac * radius`; prints overall precision / recall / F1 + radius MAE **and**
  precision/recall binned by crowding (GUVs per image) — the reference curve in
  place of a Hough baseline.

All hyperparameters live in `configs/train.yaml` and `configs/eval.yaml`.

### Real-image inference experiments

The trained model lives at **`models/best.pt`** — the default checkpoint for both
scripts below, so you never pass `--checkpoint`. (`models/` is gitignored because
the `.pt` is large; the directory is kept via `models/.gitkeep`. Set up locally
with `uv sync` first.)

Qualitatively check how the synthetic-trained detector transfers to the real
images (no ground truth — eyeball it). Output **auto-names by threshold** so runs
never overwrite each other:

```
results/
  thresh_0.10/   <- one run's PNGs (per-image 3-panel + _grid.png)
  thresh_0.15/
  ...
```

**Single threshold** (default checkpoint `models/best.pt`, images `data/`):

```bash
uv run python scripts/detect_real.py --threshold 0.15      # -> results/thresh_0.15/
uv run python scripts/detect_real.py --threshold 0.30 --nms-dist 8
```

**Sweep several thresholds in one invocation** — each to its own
`results/thresh_<value>/`:

```bash
# default sweep [0.10, 0.15, 0.20, 0.25, 0.30]
uv run python scripts/sweep_thresholds.py

# custom list / inputs
uv run python scripts/sweep_thresholds.py --thresholds 0.1 0.2 0.3 --nms-dist 8
```

Each per-image PNG is 3 panels — raw image | image + predicted circles/centers |
predicted center heatmap — titled with the filename and detection count, plus an
overview `_grid.png`. `detect_real.py` loads via the **same** `[0,1]`
normalization as training and runs the identical `detect.py` pipeline. Flags:
`--checkpoint`, `--images`, `--out` (override the auto-named folder), `--threshold`,
`--nms-dist`, `--channel`, `--limit`, `--device` (add `--device cpu` if no GPU).
`results/` is gitignored (kept via `.gitkeep`).

## Conventions

- **All sizing is in DIAMETER (pixels)** throughout the config and ground
  truth. The code converts to radius internally. Note `d_min`/`d_max` bound the
  **apparent** (image / chord) diameter — the **true sphere** diameter is
  sampled separately and is typically larger.
- All parameters live in `configs/sim_default.yaml`; no physics constants or
  paths are hardcoded in `src/`.
- Coordinates: `x` = column, `y` = row.
- Parameters tagged `[CALIBRATE]` in the config are placeholders awaiting
  calibration against real data (see step 2).

See [CLAUDE.md](CLAUDE.md) for the physics model and developer notes.
