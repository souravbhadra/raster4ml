![raster4ml-logo](https://raw.githubusercontent.com/souravbhadra/raster4ml/master/images/raster4ml_logo.png)

When geospatial raster data is concerned in a machine learning pipeline, it is often required to extract meaningful features, such as vegetation indices (e.g., NDVI, EVI, NDRE, etc.) or textures. This package provides easy-to-use functions that can automatically calculates the features with one or several lines of codes in Python. It also has the functionality of extracting statistics based on shapefile (i.e., point or polygon) from a raster data. Any type of raster data is supported regardless of satellite or UAVs.

# Key Features
- **Stacking Bands:** Stack or combine bands together to form a multiband raster data.
- **Calculate Features:** Automatically calculate necessary vegetation indices without providing any formula. Currently the package support 56 vegetation indices. (More will come)
- **Extract raster values based on shapefile:** Extract pixel values of a given image by a point shapefile or statistics from a given polygon. Also has batch-wise processign support for multiple images.
- **Clip raster based on polygon:** Clip a raster image based on the shape of given polygon file.

# How to Install?
## Dependencies
**Raster4ML** is built on top of [geopandas](https://geopandas.org/en/stable/), [rasterio](https://rasterio.readthedocs.io/en/latest/), [fiona](https://github.com/Toblerity/Fiona), [pyproj](https://pyproj4.github.io/pyproj/stable/), [rtree](https://github.com/Toblerity/rtree), [shapely](https://shapely.readthedocs.io/en/stable/manual.html), [numpy](https://numpy.org/), and [pandas](https://pandas.pydata.org/).

## Virtual Environment
It is prefered to use a virtual environment for working with this package. Use [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to create a seperate environment and then install the package and its dependencies there.
```
conda create --name raster4ml python=3
conda activate raster4ml
```

## Windows
To install on Windows, first download the wheel files for **GDAL**, **rasterio**, and **fiona** from [Christoph Gohlke's website](https://www.lfd.uci.edu/~gohlke/pythonlibs/) (🤗Thanks Christoph🤗). Go to his website, press <code>Ctrl+F</code> and type gdal. Download the GDAL file that mostly matches your computer configuration (64-bit or 32-bit) and Python version.

After downloading it, <code>cd</code> into the downloaded directory while the <code>raster4ml</code> environment is activated. Then install using <code>pip</code>. Do the same for **rasterio** and **fiona**.
```
pip install GDAL‑3.4.3‑cp310‑cp310‑win_amd64.whl
pip install rasterio‑1.2.10‑cp310‑cp310‑win_amd64.whl
pip install Fiona‑1.8.21‑cp310‑cp310‑win_amd64.whl
```
If these three are installed, the rest of the dependencies can be installed directly through **Raster4ML**'s <code>pip</code> distribution.
```
pip install raster4ml
```

## Mac OS
Has not been tested yet. 😕

## Linux
Has not been tested yet. 😕