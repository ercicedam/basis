# Inputs

Drop one or more sample optical rasters here (GeoTIFF, single
panchromatic band) to try the pipeline end-to-end.

A sample Landsat 8 panchromatic scene is published as a
[GitHub Release](https://github.com/ercicedam/basis/releases/tag/v0.1.0)
asset (it's ~312 MB, too large for git). Download it into this folder:

```bash
curl -L -o inputs/LC08_L1GT_008113_20241214_20241214_02_RT_B8_image.tif \
    https://github.com/ercicedam/basis/releases/download/v0.1.0/LC08_L1GT_008113_20241214_20241214_02_RT_B8_image.tif
```

Then, from the repository root:

```bash
python scripts/generate_masks.py \
    --input inputs/ \
    --weights weights/BASIS.weights.h5 \
    --output-dir outputs/
```

This searches `inputs/` for files matching `--glob` (default `*.tif`)
and writes one `<name>_mask.tif` per input into `outputs/`.

## Input format

- A single-band GeoTIFF (panchromatic optical image), with a valid CRS
  and geotransform.
- If the image has no-data pixels, set the raster's no-data value in its
  metadata (e.g. in QGIS) rather than leaving them as 0 or another
  in-range value — the preprocessing step relies on it to exclude those
  pixels from the normalization statistics.
