
import rasterio
import cv2
import numpy as np
from rasterio.warp import calculate_default_transform, reproject, Resampling
from . import utils

def stack_bands(image_paths, out_file):
    """Stack the images together as bands.

    Parameters
    ----------
    image_paths : list
        List of images that needs to be stacked
    out_file : src
        Output path fot the stacked image. Supports .tif.

    Returns
    -------
    None
        Nothing returns, the image is saved.

    Raises
    ------
    ValueError
        If invalid image path is given.
    """
    # Read all the individual bands
    try:
        srcs = [rasterio.open(image_path) for image_path in image_paths]
    except Exception as e:
        raise ValueError(e)
    
    # Check if the x and y are same for all the bands or not
    xy = np.array([(src.height, src.width) for src in srcs])
    # Get max x and y
    max_x = xy[:, 0].max()
    max_y = xy[:, 1].max()  

    if srcs[0].nodata is None:
        nodata_value = 0
    else:
        nodata_value = srcs[0].nodata

    # Empty array to hold stack image
    stack_img = np.zeros(shape=(len(image_paths), xy[:, 0].max(), xy[:, 1].max()))
    # Loop through each src
    for i, src in enumerate(srcs):
        x, y = src.height, src.width
        if x < max_x:
            img = src.read(1)
            img[img==nodata_value] = np.nan
            img = cv2.resize(img, (max_x, max_y), interpolation=cv2.INTER_NEAREST)
            print(f"{image_paths[i]} resized.")
            stack_img[i, :, :] = img
        else:
            img = src.read(1)
            stack_img[i, :, :] = img
    # Save
    utils.save_raster(srcs[0], stack_img, out_file,
                      driver='GTiff', width=max_y, height=max_x,
                      count=len(image_paths))
    return None 


def reproject_raster(src_image_path, dst_image_path, band=None,
                     dst_crs='EPSG:4326'):
    """Reproject the raster into a different CRS.

    Parameters
    ----------
    src_image_path : str
        Path of the image to be reprojected.
    dst_image_path : src
        Path of the destination image as reprojected.
    band : int, (Optional)
        Specify the band to reproject.
    dst_crs : str
        The destination CRS in EPSG code. For example, 'EPSG:4326', Default to 'EPSG:4326'

    Returns
    -------
    None
        Nothing returns, the image is saved.
    """
    with rasterio.open(src_image_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        
        if band is None:
            with rasterio.open(dst_image_path, 'w', **kwargs) as dst:
                for i in range(1, src.count+1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest
                    )
        else:
            with rasterio.open(dst_image_path, 'w', **kwargs) as dst:
                reproject(
                    source=rasterio.band(src, band+1),
                    destination=rasterio.band(dst, band+1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest
                )
    
    return None