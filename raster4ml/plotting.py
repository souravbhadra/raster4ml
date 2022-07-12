import os
import rasterio
import folium
import numpy as np
import numpy.ma as ma
from pyproj import Transformer

def plot_raster(image_path, bands=[1, 2, 3], **kwargs):
    
    src = rasterio.open(image_path)
    img = src.read()
    img = np.moveaxis(img, 0, 2) # Move channel to the last axis
    
    if isinstance(bands, list):
        if len(bands) != 3:
            raise ValueError("Insert only 3 integers representing the image channels.")
        else:
            img_type = 'multispectral'
    elif isinstance(bands, int):
        img_type = 'single_band'
    else:
        raise ValueError("The bands should be either an integer or list of three integers.")
    
    # Select the bands
    img = img[:, :, bands]
    
    # Convert data to float
    img = img.astype(np.float32)
    
    # Handle nodata
    if src.nodata is None:
        img[img==0.0] = np.nan
    else:
        img[img==src.nodata] = np.nan
        
    # Scale
    max_ = np.nanmax(img)
    min_ = np.nanmin(img)
    img = ((img-min_)/(max_-min_)*255).astype(np.uint8)
    
    _, ax = plt.subplots(**kwargs)
    
    ax.imshow(img)
    ax.set
    
    src.close()
    
    return img


class Map(folium.folium.Map):
    
    def __init__(self):
        super().__init__()
        
        
        
    def convert_bound_crs(self, src_crs, bound):
        
        bound_wgs = []
        
        dst_crs = 4326 # Destination crs (WGS 1984)
        
        for lat, lon in bound:
            proj = Transformer.from_crs(src_crs,
                                        dst_crs,
                                        always_xy=True)
            lon_x, lat_x = proj.transform(lon, lat)
            bound_wgs.append([lat_x, lon_x])
            
        return bound_wgs
        
        
    def add_raster(self, image_path, bands):
        
        src = rasterio.open(image_path)
        src_crs = src.crs.to_epsg()
        
        min_lon, min_lat, max_lon, max_lat = src.bounds
        bound = [[min_lat, min_lon], [max_lat, max_lon]]
        
        # See if the crs is wgs or not, unless proceed
        if src_crs != 4326: # If not WGS
            bound = self.convert_bound_crs(src_crs, bound)
        # Get center lat lon
        #cen_lat, cen_lon = np.mean(np.array(bound), axis=0)
        
        # Read src as images
        img = src.read()
        img = np.moveaxis(img, 0, 2)
        img = img[:, :, bands]
        img = img.astype(np.float32)
        img[img==0.0] = np.nan
        img = ma.masked_invalid(img)
        img_norm = (img - img.min()) / (img.max() - img.min())
        img_norm = ma.filled(img_norm, fill_value=0.0)
        mask = ma.getmask(img)
        mask = mask * 1
        mask = mask[:, :, 0]
        mask = np.abs(mask-1)
        
        # Create a RGBA image
        img_rgba = np.zeros(shape=(img.shape[0], img.shape[1], 4))
        img_rgba[:, :, :3] = img_norm
        img_rgba[:, :, 3] = mask
        
        folium.raster_layers.ImageOverlay(
            name=os.path.basename(image_path).split('.')[0],
            image=img_rgba,
            bounds=bound,
            interactive=True,
        ).add_to(self)
        folium.LayerControl().add_to(self)
        self.fit_bounds(bound)