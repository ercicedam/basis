"""Preprocessing that reproduces the normalization used at training time.

The original processing chain wrote several intermediate GeoTIFFs to disk
(one per step) and relied on a second conda environment just to stack bands
together. For a single-band (panchromatic) input, all of that reduces
exactly to the two-stage normalization implemented in ``normalize_band``
below, so this module does it directly in memory with NumPy/rasterio.
"""
import numpy as np
import rasterio


def read_band(path, band=1):
    """Read one band of a raster and return ``(data, nodata, profile)``."""
    with rasterio.open(path) as src:
        data = src.read(band).astype("float64")
        nodata = src.nodata
        profile = src.profile.copy()
    return data, nodata, profile


def normalize_band(data, nodata=None, low_pct=1, high_pct=99):
    """Percentile-stretch then min/max-normalize a single band to ``[0, 1]``.

    Equivalent to running the original data-curation percentile stretch to
    uint8 (``[0, 255]``) followed by the dataset-wide min/max normalization
    performed in ``data-preprocessing.py``, specialized to a single input
    image. Pixels equal to ``nodata`` (if given) are excluded from the
    statistics and set to 0 in the output, matching the original behaviour
    for masked pixels.
    """
    valid = np.ones_like(data, dtype=bool) if nodata is None else data != nodata
    if not np.any(valid):
        raise ValueError("Raster has no valid (non-nodata) pixels.")

    p_low, p_high = np.percentile(data[valid], [low_pct, high_pct])
    if p_high == p_low:
        raise ValueError("Degenerate percentile range; check the input image.")

    stretched = np.clip((data - p_low) / (p_high - p_low) * 255.0, 0, 255)
    stretched = np.where(valid, stretched, np.nan)

    finite = stretched[np.isfinite(stretched)]
    mn, mx = finite.min(), finite.max()
    normalized = (stretched - mn) / (mx - mn)
    normalized[~np.isfinite(normalized)] = 0.0
    return normalized.astype("float32")


def pad_to_patch(array, patch_size):
    """Zero-pad a ``(H, W[, C])`` array so ``H`` and ``W`` are multiples of
    ``patch_size``.

    Mirrors the padding used at training/evaluation time: the new size is
    always at least one extra patch bigger than the input, and the original
    data is centered inside the padded array.

    Returns ``(padded, pad_height, pad_width, height, width)`` where the
    last four values are needed to crop the prediction back to the original
    extent.
    """
    if array.ndim == 2:
        array = array[..., np.newaxis]
    height, width, channels = array.shape

    new_height = (height // patch_size + 1) * patch_size
    new_width = (width // patch_size + 1) * patch_size
    pad_height = (new_height - height) // 2
    pad_width = (new_width - width) // 2

    padded = np.zeros((new_height, new_width, channels), dtype=array.dtype)
    padded[pad_height:pad_height + height, pad_width:pad_width + width, :] = array
    return padded, pad_height, pad_width, height, width
