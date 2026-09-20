"""Sliding-window inference with Gaussian-weighted blending.

Ported from ``utils/deeplearning/utils.py`` (the ``apply`` function) in the
full training codebase.
"""
import numpy as np


def gaussian_kernel(size, mu=0, sigma=1):
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    distance = np.sqrt(x ** 2 + y ** 2)
    return np.exp(-(distance - mu) ** 2 / 2 / sigma ** 2) / np.sqrt(2 / np.pi) / sigma


def sliding_window_predict(features, model, patch_size=512, batch_size=8, n_outputs=3):
    """Run ``model`` over ``features`` using overlapping patches (50% stride),
    blended with a Gaussian window so patch borders don't create visible
    seams in the output probability map.

    ``features`` must already be padded so its height/width are each a
    multiple of ``patch_size`` (see ``preprocessing.pad_to_patch``).
    """
    height, width, _ = features.shape
    weighted_prob = np.zeros((height, width, n_outputs))
    counts = np.zeros((height, width, 1))
    weights = gaussian_kernel(patch_size)[..., np.newaxis]

    patches, rows_cols = [], []

    def flush():
        if not patches:
            return
        patch_probs = model.predict(np.array(patches), verbose=0)
        for (r, c), patch_prob in zip(rows_cols, patch_probs):
            weighted_prob[r:r + patch_size, c:c + patch_size, :] += weights * patch_prob
            counts[r:r + patch_size, c:c + patch_size, :] += weights
        patches.clear()
        rows_cols.clear()

    row = 0
    while row + patch_size <= height:
        col = 0
        while col + patch_size <= width:
            patches.append(features[row:row + patch_size, col:col + patch_size, :])
            rows_cols.append((row, col))
            if len(patches) >= batch_size:
                flush()
            col += patch_size // 2
        row += patch_size // 2
    flush()

    return weighted_prob / counts
