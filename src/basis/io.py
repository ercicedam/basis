"""Helpers to write the predicted class map back out as a GeoTIFF."""
import rasterio


def save_mask(path, mask, profile):
    """Save a 2D integer class map as a single-band GeoTIFF, reusing the
    source raster's CRS/transform so it lines up pixel-for-pixel with the
    input image."""
    out_profile = profile.copy()
    out_profile.update(driver="GTiff", dtype="uint8", count=1, compress="lzw", nodata=None)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(mask.astype("uint8"), 1)
