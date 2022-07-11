import rasterio

def save_raster(src, array, out_path, **kwargs):
    """Saves a numpy image into a geospatial tif image.

    Parameters
    ----------
    src : rasterio object
        An image object opened using rasterio.
    array : numpy nd-array
        A numpy nd-array which needs to be saved.
    out_path : str
        Output path where the tif file will be saved.

    Returns
    -------
    None
        Saves the image.
    """      
    profile = src.profile
    profile.update(**kwargs)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(array)
    return None


def get_vegetation_indices_names():
    """Returns all supported vegetation index names to calcualte.

    Returns
    -------
    list
        List of vegetation indices.
    """    
    features = ['ARI_1', 'ARI_2', 'ARVI', 'CRI_1', 'CRI_2', 'DVI',
                'EVI', 'GEMI', 'GARI', 'GCI', 'GDVI', 'GLI', 'GNDVI',
                'GOSAVI', 'GRVI', 'GSAVI', 'GVI', 'IPVI', 'LCAI',
                'MCARI', 'MNLI', 'MNDWI', 'MRENDVI', 'MRESR', 'MSR',
                'MSAVI_2', 'MTVI', 'MSI', 'NLI', 'NBR', 'NBRT_1',
                'NDBI', 'NDII', 'NDLI', 'NDMI', 'NDNI', 'NDSI',
                'NDVI', 'NDWI', 'NMDI', 'OSAVI', 'PRI', 'PSRI',
                'RENDVI', 'RDVI', 'SR_1', 'SAVI', 'SIPI', 'TCARI',
                'TDVI', 'TVI', 'VARI', 'VREI_1', 'VREI_2', 'WBI',
                'WDRVI']
    return features


def check_projection(src, shape):
    """Check if the raster and shapefile has same projection or not.
    If not, it reprojects the shapefile.

    Parameters
    ----------
    src : rasterio object
        An image opened by rasterio.
    shape : geopandas object
        A shapfile opened using geopandas.

    Returns
    -------
    geopandas object
        The same given shapefile but reprojected to raster data's
        projection.
    """    
    raster_epsg = src.crs.to_epsg()
    shape_epsg = shape.crs.to_epsg()
    if shape_epsg != raster_epsg:
        shape.set_crs(epsg=raster_epsg)
    return shape    

