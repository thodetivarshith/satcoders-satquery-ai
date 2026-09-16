# GeoTIFF loader - Kunchala
import rasterio
def load_geotiff(file_path):
    """
    load a GeoTIFF image.
    Args:
        filepath: path to the GeoTIFF file.
    Returns:
        image_data: the image data as a numpy array
        metadata: the metadata about the image .
    """
    with rasterio.open(file_path) as src:
        image_data = src.read()
        metadata = {
            'width': src.width,
            'height': src.height,
            'count': src.count,
            'crs': src.crs,
            'transform': src.transform,
         }
    return image_data, metadata

