import rasterio
import numpy as np
import matplotlib.pyplot as plt

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