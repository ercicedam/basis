# BASIS — BAsal and Surface Ice-shelf fracture extraction System

BASIS is a lightweight, inference-only release of a deep learning model that segments crevasses/fractures in optical satellite imagery of glaciers and ice shelves. It is a companion to our paper (see [Citation](#citation)) and lets you generate fracture masks from your own optical rasters using the trained model, without the full training pipeline, data-curation scripts, or multi-environment setup of the [research repository](#relationship-to-the-full-repository).

The model is a DeepLabv3+-style architecture (ASPP decoder on a ResNet-50 encoder) adapted from [this DeepLabv3 walkthrough](https://medium.com/@itberrios6/deeplabv3-c0c8c93d25a4). It was developed as part of the IceDaM project, funded by the European Research Council.

## Contents

```
BASIS_code/
├── src/basis/          # model, pre-/post-processing, CLI
├── scripts/             # runnable entry point (no install needed)
├── weights/              # download the released weights here (see weights/README.md)
├── inputs/                # optional sample/your own input rasters (see inputs/README.md)
├── outputs/                # generated masks land here
├── environment.yml          # conda environment
└── environment.txt           # pip-only dependency list
```

## Installation

Using conda (recommended):

```bash
conda env create -f environment.yml
conda activate basis
```

Or with pip only, in your own Python 3.9+ environment:

```bash
pip install -r environment.txt
```

No package installation step is needed beyond the dependencies above — run `python scripts/generate_masks.py`, which adds `src/` to the path itself.

GPU inference needs a CUDA/cuDNN setup matching your driver; see the [TensorFlow install guide](https://www.tensorflow.org/install/pip). CPU inference works out of the box, just slower.

## Getting the weights and sample data

The weights file and the sample input raster are too large for git (9.5 MB and 312 MB) and are published as [GitHub Release v0.1.0](https://github.com/ercicedam/basis/releases/tag/v0.1.0) assets instead of being committed:

```bash
curl -L -o weights/BASIS.weights.h5 \
    https://github.com/ercicedam/basis/releases/download/v0.1.0/BASIS.weights.h5

curl -L -o inputs/LC08_L1GT_008113_20241214_20241214_02_RT_B8_image.tif \
    https://github.com/ercicedam/basis/releases/download/v0.1.0/LC08_L1GT_008113_20241214_20241214_02_RT_B8_image.tif
```

This repository is currently **private**, so the `curl` commands above only work once authenticated. Until it's made public, use the GitHub CLI instead: `gh release download v0.1.0 --repo ercicedam/basis`. See `weights/README.md` and `inputs/README.md` for details.

## Usage

```bash
python scripts/generate_masks.py \
    --input path/to/image.tif \
    --weights weights/BASIS.weights.h5 \
    --output-dir outputs/
```

`--input` accepts one or more files and/or directories (directories are searched with `--glob`, default `*.tif`):

```bash
python scripts/generate_masks.py \
    --input path/to/rasters_dir/ \
    --weights weights/BASIS.weights.h5 \
    --output-dir outputs/
```

Each input `<name>.tif` produces `outputs/<name>_mask.tif`, a single-band GeoTIFF of class indices (0 = background) aligned pixel-for-pixel with the input.

### Options

| Flag             | Default | Meaning                                                        |
|------------------|---------|-----------------------------------------------------------------|
| `--patch-size`   | 512     | Sliding-window patch size; must match the weights.               |
| `--n-classes`    | 3       | Number of output classes; must match the weights.                |
| `--n-features`   | 1       | Number of input bands (1 = panchromatic); must match the weights. |
| `--kernel-size`  | 1       | ASPP convolution kernel size; must match the weights.             |
| `--batch-size`   | 8       | Patches per prediction batch (lower if you hit an OOM error).     |
| `--cpu`          | off     | Force CPU inference even if a GPU is visible.                    |

The defaults match the weights configuration recommended in `weights/README.md`; only change them if you are using different weights.

## Input data requirements

- A single-band GeoTIFF (panchromatic optical image) with a valid CRS and geotransform.
- If the image has no-data pixels, set the raster's no-data value in its metadata (e.g. in QGIS). Preprocessing excludes those pixels from the normalization statistics and leaves them as background in the output.

## How it works

1. **Normalize**: the input band is percentile-stretched (1st/99th percentile) and min/max-normalized to `[0, 1]`, matching the normalization used at training time (`src/basis/preprocessing.py`).
2. **Pad**: the image is zero-padded so its height/width are multiples of `--patch-size`, centering the original data.
3. **Predict**: the model is applied over overlapping patches (50% stride) and predictions are blended with a Gaussian window to avoid seams at patch borders (`src/basis/inference.py`).
4. **Save**: the per-pixel class with the highest probability is written out as a GeoTIFF, cropped back to the original extent.

## Relationship to the full repository

This release only ships the mask-generation path of the full research project (developed under the working name CrevaU-Net). It intentionally leaves out what isn't needed to run a trained model on new images: the training pipeline, shapefile-driven data curation, HDF5 dataset preparation, and the two-conda-environment setup. The normalization and sliding-window inference are re-implemented here to be numerically equivalent to the originals, without needing a second environment or writing intermediate files to disk. If you want to retrain or fine-tune the model, or need the full data-curation pipeline, see the main research repository.

## Citation

If you use this code or the released weights, please cite:

> TODO: paper reference.

## License

> TODO

## Acknowledgments

Developed under the IceDaM project, funded by the European Research Council (ERC), at the Institut des Géosciences de l'Environnement (IGE), Université Grenoble Alpes.

## Contact

- Kaian Shahateet — kaian@ita.br
- Colin Prieur — colin.prieur@univ-grenoble-alpes.fr
- Romain Millan — romain.millan@univ-grenoble-alpes.fr
