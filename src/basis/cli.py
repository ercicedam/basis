"""Command-line interface to generate fracture masks from optical rasters.

Example
-------
    python -m basis.cli \\
        --input path/to/image.tif \\
        --weights weights/DeepLabMini.weights.h5 \\
        --output-dir outputs/
"""
import argparse
import glob
import os
import shutil
import sys
import tempfile

import numpy as np

from .inference import sliding_window_predict
from .io import save_mask
from .model import build_deeplab_mini
from .preprocessing import normalize_band, pad_to_patch, read_band


def _load_weights(model, weights_path):
    """Load a ``.h5`` weights file, tolerating the legacy TF/Keras <= 2.12
    HDF5 layout that the released weights use.

    Keras dispatches the loader purely from the filename: paths ending in
    ``.weights.h5`` are assumed to use the newer native format (variables
    stored under a per-layer ``vars`` group), while any other ``.h5``/
    ``.hdf5`` path uses the legacy per-layer format. If a legacy-format
    file happens to be named ``*.weights.h5``, Keras picks the wrong
    loader and raises ``KeyError: ... 'vars' doesn't exist``. Work around
    it by retrying from a copy whose name doesn't end in ``.weights.h5``.
    """
    try:
        model.load_weights(weights_path)
    except KeyError as err:
        if "vars" not in str(err):
            raise
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, "legacy_weights.h5")
            shutil.copyfile(weights_path, tmp_path)
            model.load_weights(tmp_path)


def _collect_inputs(input_args, pattern):
    paths = []
    for item in input_args:
        if os.path.isdir(item):
            paths.extend(sorted(glob.glob(os.path.join(item, pattern))))
        else:
            paths.append(item)
    if not paths:
        raise FileNotFoundError(
            f"No input rasters found (looked for '{pattern}' inside any given directories)."
        )
    return paths


def generate_mask(image_path, model, patch_size, batch_size, n_classes):
    """Run the full pipeline (read -> normalize -> pad -> predict -> crop)
    for a single raster and return ``(mask, profile)``."""
    data, nodata, profile = read_band(image_path)
    normalized = normalize_band(data, nodata=nodata)
    padded, pad_h, pad_w, height, width = pad_to_patch(normalized, patch_size)

    prob = sliding_window_predict(
        padded, model, patch_size=patch_size, batch_size=batch_size, n_outputs=n_classes
    )
    mask = np.argmax(prob, axis=-1)
    mask = mask[pad_h:pad_h + height, pad_w:pad_w + width]
    return mask, profile


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="One or more raster files, and/or directories to search (see --glob).",
    )
    parser.add_argument(
        "--glob", default="*.tif",
        help="Filename pattern used when --input contains a directory (default: *.tif).",
    )
    parser.add_argument("--weights", required=True, help="Path to the DeepLabMini .weights.h5 file.")
    parser.add_argument("--output-dir", required=True, help="Directory where masks are written.")
    parser.add_argument(
        "--patch-size", type=int, default=512,
        help="Must match the value used to train --weights (default: 512).",
    )
    parser.add_argument(
        "--n-classes", type=int, default=3,
        help="Number of output classes; must match --weights (default: 3).",
    )
    parser.add_argument(
        "--n-features", type=int, default=1,
        help="Number of input bands; must match --weights (default: 1, panchromatic).",
    )
    parser.add_argument(
        "--kernel-size", type=int, default=1,
        help="ASPP convolution kernel size; must match --weights (default: 1).",
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Patches per prediction batch.")
    parser.add_argument("--cpu", action="store_true", help="Force inference on CPU even if a GPU is available.")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    if args.cpu:
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")

    os.makedirs(args.output_dir, exist_ok=True)
    input_paths = _collect_inputs(args.input, args.glob)

    input_shape = (args.patch_size, args.patch_size, args.n_features)
    model = build_deeplab_mini(input_shape, args.n_classes, kernel_size=args.kernel_size)
    _load_weights(model, args.weights)

    for path in input_paths:
        print(f"Processing {path}")
        mask, profile = generate_mask(path, model, args.patch_size, args.batch_size, args.n_classes)
        out_path = os.path.join(
            args.output_dir, os.path.splitext(os.path.basename(path))[0] + "_mask.tif"
        )
        save_mask(out_path, mask, profile)
        print(f"  -> {out_path}")


if __name__ == "__main__":
    sys.exit(main())
