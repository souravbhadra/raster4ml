import os
import rasterio
import folium
import ipyleaflet
import rioxarray
import geopandas as gpd
import numpy as np
import numpy.ma as ma
from pyproj import Transformer
import matplotlib.pyplot as plt

class Map(folium.folium.Map):
    
    """A Folium map objecto to make an interactive map.

    Attributes
    ----------
    kwargs : dict
        Keywords arguments of folium.folum.Map.

    Methods
    -------
    add_raster(image_path, bands, layer_control=True):
        Add a raster data to the map. Note: layer_control must be False if another feature
        is to be added later. It should be True, if this is the only or last feature to
        add.
    """
    
    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)
        
        
    def convert_bound_crs(self, src_crs, bound):
        """Convert the bound from any CRS to WGS 1984 CRS.

        Parameters
        ----------
        src_crs : int
            The source CRS from EPSG.
        bound : list
            The bound in [[min_lat, min_lon], [max_lat, max_lon]].

        Returns
        -------
        list
            The converted bound in WGS format.
        """ 
        
        bound_wgs = []
        
        dst_crs = 4326 # Destination crs (WGS 1984)
        
        for lat, lon in bound:
            proj = Transformer.from_crs(src_crs,
                                        dst_crs,
                                        always_xy=True)
            lon_x, lat_x = proj.transform(lon, lat)
            bound_wgs.append([lat_x, lon_x])
            
        return bound_wgs
    
    
    def colorize_band(self, array, cmap='viridis'):
        """Colorize a single band to a RGBA image.

        Parameters
        ----------
        array : numpy nd-array
            The array of the band.
        cmap : str
            The colormap to be drawn, supports matplotlib cmas, default to 'viridis'.

        Returns
        -------
        numpy nd-array
            A colorized 4-band RGBA array.
        """ 
        normed_data = (array - array.min()) / (array.max() - array.min())    
        cm = plt.cm.get_cmap(cmap)    
        return cm(normed_data)
        
        
    def add_raster(self, image_path, bands=None, layer_control=False):
        """Add a raster data into the map.

        Parameters
        ----------
        image_path : str
            The path of the image.
        bands : list of int
            If multispectral, then provide the three bands to use as RGB, Defaults to 
            [3, 2, 1]. If single band image, Defaults to the first band (bands = 0). If
            int is given, then only one bands will be drawn.
        layer_control : bool
            If True, then the layer control will be provided on the map. It should be 
            False if another feature is to be drawn after this feature draw. Defaults to
            True.

        Returns
        -------
        None
        """ 
        
        src = rasterio.open(image_path)
        src_crs = src.crs.to_epsg()
        
        min_lon, min_lat, max_lon, max_lat = src.bounds
        bound = [[min_lat, min_lon], [max_lat, max_lon]]
        
        # See if the crs is wgs or not, unless proceed
        if src_crs != 4326: # If not WGS
            bound = self.convert_bound_crs(src_crs, bound)
        # Get center lat lon
        #cen_lat, cen_lon = np.mean(np.array(bound), axis=0)
        
        # Check band
        if src.count > 1: # If multi-band
            if bands is None:
                bands = [3, 2, 1]
        else:
            bands = 0
        
        # Read src as images
        img = src.read()
        img = np.moveaxis(img, 0, 2)
        img = img[:, :, bands]
        img = img.astype(np.float32)
        if src.nodata == None:
            img[img==0.0] = np.nan
        else:
            img[img==src.nodata] = np.nan
        img = ma.masked_invalid(img)
        
        # Create a RGBA image
        if src.count > 1:
            img_norm = (img - img.min()) / (img.max() - img.min())
            img_norm = ma.filled(img_norm, fill_value=0.0)
            mask = ma.getmask(img)
            mask = mask * 1
            mask = mask[:, :, 0]
            mask = np.abs(mask-1)
            img_rgba = np.zeros(shape=(img.shape[0], img.shape[1], 4))
            img_rgba[:, :, :3] = img_norm
            img_rgba[:, :, 3] = mask
        else:
            img_rgba = self.colorize_band(img)
        
        folium.raster_layers.ImageOverlay(
            name=os.path.basename(image_path),
            image=img_rgba,
            bounds=bound,
            interactive=True,
        ).add_to(self)
        self.fit_bounds(bound)
        if layer_control:
            folium.LayerControl().add_to(self)
        
        
    def add_shape(self, shape_path, layer_control=False):
        """Add a shapefile data into the map.

        Parameters
        ----------
        shape_path : str
            The path of the shapefile.
        layer_control : bool
            If True, then the layer control will be provided on the map. It should be 
            False if another feature is to be drawn after this feature draw. Defaults to
            True.

        Returns
        -------
        None
        """ 
        
        shape = gpd.read_file(shape_path)
        
        if shape.crs.to_epsg() != 4326:
            shape = shape.to_crs(epsg=4326)
        
        shape_json = shape.to_json()
        folium.features.GeoJson(
            data=shape_json,
            name=os.path.basename(shape_path)
        ).add_to(self)
        
        min_lon, min_lat, max_lon, max_lat = shape.total_bounds
        bound = [[min_lat, min_lon], [max_lat, max_lon]]
        self.fit_bounds(bound)
        
        if layer_control:
            folium.LayerControl().add_to(self)
        