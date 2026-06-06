"""Shared image normalization -- the SINGLE place a uint8 image becomes [0, 1].

Both the training data loader (`src/dataset.py`) and inference (`src/detect.py`)
import and use these functions, so the preprocessing applied to synthetic
training images and to real images at test time is guaranteed identical and can
never drift apart. The synthetic dataset is generated as uint8 PNG specifically
so it matches the real 512x512 uint8 images bit-for-bit through this path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

UINT8_MAX = 255.0


def to_unit(img: np.ndarray) -> np.ndarray:
    """Convert a raw (uint8) image array to float32 in [0, 1] by dividing by 255.

    This single operation is the normalization contract shared by training and
    inference. Do NOT divide by `img.max()` or anything data-dependent -- the
    scale must be fixed so train and test see the same mapping.
    """
    return np.asarray(img, dtype=np.float32) / UINT8_MAX


def _read_raw(path: str | Path, channel: int = 1) -> np.ndarray:
    """Read an image file to a 2D uint8-scale array (handles PNG and TIFF).

    Real lipid images are single-channel 512x512; if a multi-channel stack is
    passed, `channel` selects the lipid (561 nm) channel.
    """
    path = Path(path)
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        arr = np.asarray(tifffile.imread(path))
    else:
        from PIL import Image

        arr = np.asarray(Image.open(path))

    if arr.ndim == 3:
        # Channel-first (C,H,W) if a small leading axis, else channel-last.
        if arr.shape[0] <= 4:
            arr = arr[channel]
        elif arr.shape[-1] <= 4:
            arr = arr[..., channel]
        else:
            raise ValueError(f"unsupported image shape {arr.shape} for {path}")
    return arr


def load_normalized(path: str | Path, channel: int = 1) -> np.ndarray:
    """Load any supported image as float32 in [0, 1], shape (H, W).

    Used by BOTH the training dataset and detect.py so synthetic and real images
    are preprocessed identically.
    """
    return to_unit(_read_raw(path, channel=channel))
