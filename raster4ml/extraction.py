import os
import glob
import rasterio
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.mask import mask
from tqdm import tqdm
from . import utils


def extract_by_points(image_path,
                      shape_path,
                      unique_id):
    """Extract value by a point shapefile.
    
    The function will extract the pixel values underneath each point
    in a given point shapefile.

    Parameters
    ----------
    image_path : str
        Path of the image.
    shape_path : str
        Path of the shapefile.
    unique_id : str
        A unique column in the shapefile which will be retained as
        the id.

    Returns
    -------
    pandas dataframe
        A pandas dataframe containing all the values per id.

    Raises
    ------
    ValueError
        The shapefile must be either Point or MultiPoint.
    """
    # Open files
    src = rasterio.open(image_path)
    img = src.read(1)
    shape = gpd.read_file(shape_path)

    # Check the geometry types of the shape
    geoms = shape.geom_type.values
    if all((geoms == 'Point') | (geoms == 'MultiPoint')):
        pass
    else:
        raise ValueError(
            "The shapefile must be either Point or MultiPoint.")

    # Image names
    image_name = os.path.basename(image_path).split('.')[0]

    # Check if the CRS of shapefile and raster data matches or not
    shape = utils.check_projection(src, shape)

    # Check the type of geometry in shape

    pixel_values = {}
    for i, point in enumerate(shape['geometry']):
        x = point.xy[0][0]
        y = point.xy[1][0]
        row, col = src.index(x, y)
        value = img[row, col]
        pixel_values[shape.loc[i, f'{unique_id}']] = [value]
    pixel_values = pd.DataFrame(pixel_values).T
    pixel_values.columns = [image_name]
    # Close the rasterio
    src.close()
    return pixel_values


def get_duplicated_columns(df):
    """Get the columns which has values all the same.

    Parameters
    ----------
    df : pandas DataFrame
        The pandas dataframe to check.   

    Returns
    -------
    list
        Lis of columns that needs to be removed.
    """    
    df = df.fillna(1, axis=1) # Replace all nans with 1
    duplicates = []
    for col in df.columns:
        if (df[col] == df[col][0]).all():
            duplicates.append(col)
    return duplicates

def batch_extract_by_points(image_paths,
                            shape_path,
                            unique_id):
    """Batch extract values from a set of raster data using point
    shapefile.

    Parameters
    ----------
    image_paths : list
        List of image paths.
    shape_path : str
        Path of the shapefile.
    unique_id : str
        A unique column in the shapefile which will be retained as
        the id.   

    Returns
    -------
    pandas dataframe
        A pandas dataframe containing all the values per id.
    """
    pixel_values_df = []
    for image_path in tqdm(image_paths):
        pixel_values = extract_by_points(image_path, shape_path, unique_id)
        pixel_values_df.append(pixel_values)
    pixel_values_df = pd.concat(pixel_values_df, axis=1)
    
    # Check if there are duplicate columns
    cols_to_remove = get_duplicated_columns(pixel_values_df)
    if len(cols_to_remove) > 0:
        pixel_values_df = pixel_values_df.drop(columns=cols_to_remove)
        print(f'{len(cols_to_remove)} columns were removed from the dataframe as they had duplicated values.')
    return pixel_values_df


def extract_by_polygons(image_path,
                        shape_path,
                        unique_id,
                        statistics='all',
                        prefix=None):
    """Extract value from a raster data using a polygon shapefile.
    Similar to Zonal Statistics.

    Parameters
    ----------
    image_path : str
        Path of the image.
    shape_path : str
        Path of the shapefile.
    unique_id : str
        A unique column in the shapefile which will be retained as
        the id.
    statistics : str or list
        List of statistics to be calculated if shape is polygon.
        Accepted statsitcs are either 'all' or a list containing 
        follwoing statistics:
        'mean', 'median', 'mode', 'sum', 'min', 'max', 'std', 'range',
        'iqr', 'unique'.
        If only one statistic to be calculated, that should be inside
        a list. For example, if only 'mean' is to be calculated, it
        should be given as ['mean'].
    prefix : str, optional
        If predix is given, then the prefix will be used in front of
        the statistics name within the final dataframe column,
        by default None

    Returns
    -------
    pandas dataframe
        A pandas dataframe containing all the statistics values per
        id.

    Raises
    ------
    ValueError
        The shapefile must be either Polygon or MultiPolygon.
    """
    # Open files
    src = rasterio.open(image_path)
    shape = gpd.read_file(shape_path)

    # Check the geometry types of the shape
    geoms = shape.geom_type.values
    if all((geoms == 'Polygon') | (geoms == 'MultiPolygon')):
        pass
    else:
        raise ValueError(
            "The shapefile must be either Polygon or MultiPolygon.")

    # Check if the CRS of shapefile and raster data matches or not
    shape = utils.check_projection(src, shape)

    stats = {}

    for i, polygon in enumerate(shape['geometry']):
        mask_img, _ = mask(src, [polygon], nodata=np.nan, crop=True)
        mask_img = mask_img.reshape(-1)

        if statistics == 'all':
            statistics = ['mean', 'median', 'sum', 'min',
                          'max', 'std', 'range', 'iqr', 'unique'] # removed mode

        # Check if all values are nan
        if np.isnan(mask_img).all():
            stats_values = [np.nan]*len(statistics)
        else:
            mask_img = mask_img[~np.isnan(mask_img)]
            stats_values = []
            if 'mean' in statistics:
                stats_values.append(np.mean(mask_img))
            if 'median' in statistics:
                stats_values.append(np.median(mask_img))
            #if 'mode' in statistics:
            #    stats_values.append(np.bincount(mask_img).argmax())
            if 'sum' in statistics:
                stats_values.append(np.sum(mask_img))
            if 'min' in statistics:
                stats_values.append(np.min(mask_img))
            if 'max' in statistics:
                stats_values.append(np.max(mask_img))
            if 'std' in statistics:
                stats_values.append(np.std(mask_img))
            if 'range' in statistics:
                stats_values.append(np.max(mask_img)-np.min(mask_img))
            if 'iqr' in statistics:
                stats_values.append(np.subtract(
                    *np.percentile(mask_img, [75, 25])))
            if 'unique' in statistics:
                stats_values.append(np.unique(mask_img).shape[0])

        stats[shape.loc[i, f'{unique_id}']] = stats_values

    stats = pd.DataFrame(stats).T
    if prefix is None:
        stats.columns = statistics
    else:
        stats.columns = [
            f'{str(prefix)}_{statistic}' for statistic in statistics]

    # Close rasterio
    src.close()

    return stats


def batch_extract_by_polygons(image_paths,
                              shape_path,
                              unique_id,
                              statistics='all'):
    """Batch extract value by a polygon shapefile from a given image
    paths. Similar to zonal statistics.

    Parameters
    ----------
    image_paths : list
        List of image paths.
    shape_path : str
        Path of the shapefile.
    unique_id : str
        A unique column in the shapefile which will be retained as
        the id.
    statistics : str or list
        List of statistics to be calculated if shape is polygon.
        Accepted statsitcs are either 'all' or a list containing 
        follwoing statistics:
        'mean', 'median', 'mode', 'sum', 'min', 'max', 'std', 'range',
        'iqr', 'unique'.
        If only one statistic to be calculated, that should be inside
        a list. For example, if only 'mean' is to be calculated, it
        should be given as ['mean'].
    prefix : str, optional
        If predix is given, then the prefix will be used in front of
        the statistics name within the final dataframe column,
        by default None 

    Returns
    -------
    pandas dataframe
        A pandas dataframe containing all the statistics values per
        id. Each column name will be made through automatically adding
        a prefix (which is the filename of each image) and the
        corresponding statistics.
    """
    stats_df = []
    image_paths = glob.glob(os.path.join(image_paths, "*.tif"))
    for image_path in tqdm(image_paths):
        prefix = os.path.basename(image_path).split('.')[0]
        stats = extract_by_polygons(image_path,
                                    shape_path,
                                    unique_id,
                                    statistics, 
                                    prefix=prefix)
        stats_df.append(stats)
    stats_df = pd.concat(stats_df, axis=1)

    # Check if there are duplicate columns
    cols_to_remove = get_duplicated_columns(stats_df)
    if len(cols_to_remove) > 0:
        stats_df = stats_df.drop(columns=cols_to_remove)
        print(f'{len(cols_to_remove)} columns were removed from the dataframe as they had duplicated values.')
    return stats_df


def clip_by_polygons(image_path,
                     shape_path,
                     unique_id,
                     out_dir,
                     out_type='numpy'):
    """Clip a raster image by a polygon shapefile.

    Based on the geometry of each polygon, the function will clip the
    images and save it in a given directory with a unique name.

    Parameters
    ----------
    image_path : str
        Path of the image.
    shape_path : str
        Path of the shapefile.
    unique_id : str
        A unique column in the shapefile which will be retained as
        the id.
    out_dir : str
        Path of the directory where the clipped images will be saved.
    out_type : str, optional
        The type of output data. It can be either 'numpy' or 'tif',
        by default 'numpy'

    Returns
    -------
    None
        Saves the clipped data.

    Raises
    ------
    ValueError
        The shapefile must be either Polygon or MultiPolygon.
    """
    # Open files
    src = rasterio.open(image_path)
    shape = gpd.read_file(shape_path)

    # Check the geometry types of the shape
    geoms = shape.geom_type.values
    if all((geoms == 'Polygon') | (geoms == 'MultiPolygon')):
        pass
    else:
        raise ValueError(
            "The shapefile must be either Polygon or MultiPolygon.")

    # Check if the CRS of shapefile and raster data matches or not
    shape = utils.check_projection(src, shape)

    for i, polygon in enumerate(shape['geometry']):
        mask_img, transform = mask(src, [polygon], nodata=0, crop=True)
        if out_type == 'numpy':
            out_path = os.path.join(out_dir, str(
                shape.loc[i, f'{unique_id}'])+'.npy')
            mask_img = np.moveaxis(mask_img, 0, 2)
            np.save(out_path, mask_img)
        elif out_type == 'tif':
            out_path = os.path.join(out_dir, str(
                shape.loc[i, f'{unique_id}'])+'.tif')
            utils.save_raster(src, mask_img, out_path,
                              driver='GTiff', nodata=0,
                              width=mask_img.shape[2],
                              height=mask_img.shape[1],
                              transform=transform)
    return None
