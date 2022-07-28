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
    features = [
        'ATSAVI', 'AFRI1600', 'AFRI2100', 'ARI', 'AVI', 'ARVI', 'ARVI2',
        'BWDRVI', 'BRI', 'CCCI', 'CASI_NDVI', 'CASI_TM4_3', 'CARI', 'CARI2',
        'Chlgreen', 'CIgreen', 'CIrededge710', 'CIrededge', 'Chlrededge',
        'CVI', 'CI', 'CTVI', 'CRI550', 'CRI700', 'CI', 'Datt1', 'Datt4',
        'Datt6', 'D678_500', 'D800_550', 'D800_680', 'D833_658', 'GDVI', 
        'DVIMSS', 'DSWI', 'DSWI5', 'DD', 'DPI', 'EVI', 'EVI2', 'EVI3',
        'Gitelson', 'GEMI', 'GVMI', 'GARI', 'GLI', 'GNDVI', 'GOSAVI', 'GSAVI',
        'GBNDVI', 'Hue', 'PVIhyp', 'IPVI', 'Intensity', 'IR550', 'IR700',
        'LCI', 'LWCI', 'LogR', 'Maccioni', 'mCRIG', 'mCRIRE', 'MTCI', 'MVI',
        'MGVI', 'MNSI', 'MSBI', 'MYVI', 'mND680', 'mARI', 'MCARI', 'MCARI1',
        'MCARI1510', 'MCARI2', 'MCARI705', 'MCARI710', 'mNDVI', 'Vog2',
        'MND_750_705', 'MND_734_747_715_720', 'MND_850_1788_1928',
        'MND_850_2218_1928', 'MRVI', 'mSR', 'MSR670', 'MSR705', 'MSR_705_445',
        'MSR_NIR_Red', 'MSAVI', 'MSAVIhyper', 'MTVI1', 'MTVI2', 'MNLI',
        'mSR2', 'DDn', 'NLI', 'NormG', 'NormNIR', 'NormR', 'NDWI_Hyp',
        'ND_1080_1180', 'ND_1080_1260', 'ND_1080_1450', 'ND_1080_1675',
        'ND_1080_2170', 'LWVI1', 'LWVI2', 'ND_1180_1450', 'ND_1180_1675',
        'ND_1180_2170', 'ND_1260_1450', 'ND_1260_1675', 'ND_1260_2170',
        'ND_1510_660', 'NDBleaf', 'NDlma', 'NPQI', 'PRI_528_587', 
        'PRI_531_570', 'PPR', 'PRI_550_530', 'PVR', 'PRI_570_539', 
        'PRI_570_531', 'NPCI', 'ND_682_553', 'NDVIg', 'ND_750_650',
        'ND_750_660', 'ND_750_680', 'NDVI_705', 'RENDVI', 'ND_774_677',
        'GNDVIhyper', 'ND_782_666', 'ND_790_670', 'NDRE', 'ND_800_1180',
        'ND_800_1260', 'ND_800_1450', 'ND_800_1675', 'ND_800_2170', 
        'PSNDc2', 'PSNDc1', 'GNDVIhyper2', 'PSNDb2', 'PSNDb1', 'PSNDa1',
        'ND_800_680', 'ND_819_1600', 'ND_819_1649', 'NDMI', 'ND_827_668',
        'ND_833_1649', 'ND_833_658', 'ND_850_1650', 'ND_857_1241',
        'ND_860_1240', 'ND_895_675', 'ND_900_680', 'NDchl', 'ND_960_1180',
        'ND_960_1260', 'ND_960_1450', 'ND_960_1675', 'ND_960_2170', 
        'NGRDI', 'NDLI', 'NDVI', 'BNDVI', 'GNDVI', 'NDRE', 'NBR', 'NDNI', 
        'RI', 'NDVI_rededge', 'NDSI', 'NDVI_700', 'OSAVI', 'OSAVI_1510', 
        'OSAVI2', 'PNDVI', 'PSRI', 'R_675_700_650', 'R_WI_ND750', 'RDVI', 
        'RDVI2', 'Rededge1', 'Rededge2', 'RBNDVI', 'REIP1', 'REIP2', 
        'REIP3', 'REP', 'RVSI', 'RRE', 'RDVI', 'SAVImir', 'IF', 
        'SR_1058_1148', 'SR_1080_1180', 'SR_1080_1260', 'SR_1080_1450', 
        'SR_1080_1675', 'SR_1080_2170', 'SR_1180_1080', 'SR_1180_1450', 
        'SR_1180_1675', 'SR_1180_2170', 'SR_1193_1126', 'SR_1250_1050', 
        'SR_1260_1080', 'SR_1260_1450', 'SR_1260_1675', 'SR_1260_2170', 
        'SR_1450_1080', 'SR_1450_1180', 'SR_1450_1260', 'SR_1450_960', 
        'SR_1600_820', 'SR_1650_2218', 'SR_1660_550', 'SR_1660_680', 
        'SR_1675_1080', 'SR_1675_1180', 'SR_1675_1260', 'SR_1675_960', 
        'SR_2170_1080', 'SR_2170_1180', 'SR_2170_1260', 'SR_2170_960', 
        'SR_430_680', 'SR_440_690', 'SR_440_740', 'BGI', 'BRI', 
        'SR_520_420', 'SR_520_670', 'SR_520_760', 'SR_542_750', 
        'SR_550_420', 'SR_550_670', 'SR_550_680', 'SR_550_760', 
        'SR_550_800', 'SR_554_677', 'SR_556_750', 'SR_560_658', 
        'SR_605_420', 'SR_605_670', 'SR_605_760', 'SR_672_550', 
        'SR_672_708', 'SR_674_553', 'SR_675_555', 'SR_675_700', 
        'SR_678_750', 'SR_683_510', 'SR_685_735', 'SR_690_735', 
        'SR_690_740', 'SR_694_840', 'SR_695_420', 'SR_695_670', 
        'SR_695_760', 'SR_695_800', 'SR_700', 'SR_700_670', 
        'SR_705_722', 'SR_706_750', 'SR_710_420', 'SR_710_670', 
        'SR_710_760', 'SR_715_705', 'SR_730_706', 'SR_735_710', 
        'SR_740_720', 'SR_750_550', 'SR_750_555', 'SR_750_700', 
        'SR_750_705', 'SR_750_710', 'SR_752_690', 'Datt3', 
        'RARS', 'SR_760_695', 'SR_774_677', 'SR_800_1180', 
        'SR_800_1280', 'SR_800_1450', 'SR_800_1660', 
        'SR_800_1675', 'SR_800_2170', 'SR_800_470', 
        'SR_800_500', 'SR_800_550', 'SR_800_600', 'SR_800_635',
        'SR_800_650', 'SR_800_670', 'SR_800_675', 'SR_800_680',
        'SR_800_960', 'SR_800_550', 'SR_800_670', 'SR_810_560',
        'SR_833_1649', 'SR_833_658', 'SR_850_710', 'SR_850_1240',
        'SR_850_550', 'SR_850_708', 'SR_895_972', 'SR_900_680',
        'SR_950_900', 'SR_960_1180', 'SR_960_1260', 'SR_960_1450',
        'SR_960_1675', 'SR_960_2170', 'PWI', 'SR_970_902', 'SRPI',
        'SR_355_365_gk', 'SAVI', 'SARVI2', 'SAVI3', 'SBL', 'SLAVI', 
        'SPVI', 'SQRT_NIR_R', 'SIPI1', 'SIPI2', 'SIPI3', 'SBI',
        'GVIMSS', 'NSIMSS', 'SBIMSS', 'GVI', 'WET', 'YVIMSS',
        'TCARI', 'TCARI1510', 'TCARI2', 'TNDVI', 'TSAVI', 'TVI',
        'TCI', 'TGI', 'TVI', 'VARIgreen', 'VARI700', 'VARIrededge',
        'WDRVI'
    ]
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

