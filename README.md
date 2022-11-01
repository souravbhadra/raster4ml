![check-status](https://img.shields.io/github/checks-status/remotesensinglab/raster4ml/master)
![docs](https://img.shields.io/readthedocs/raster4ml)
![license](https://img.shields.io/github/license/remotesensinglab/raster4ml)
![downloads](https://img.shields.io/github/downloads/remotesensinglab/raster4ml/total)



![raster4ml-logo](https://raw.githubusercontent.com/souravbhadra/raster4ml/master/docs/images/raster4ml_logo.png)

When geospatial raster data is concerned in a machine learning pipeline, it is often required to extract meaningful features, such as vegetation indices (e.g., NDVI, EVI, NDRE, etc.) or textures. This package provides easy-to-use functions that can automatically calculates the features with one or several lines of codes in Python. It also has the functionality of extracting statistics based on shapefile (i.e., point or polygon) from a raster data. Any type of raster data is supported regardless of satellite or UAVs.

## Key Features
- **Stack raster bands**
- **Automatically calculate vegetation indices (supports 350+ indices)**
- **Extract raster values based on shapefile**
- **Clip raster based on polygon**


## Documentation
Detailed documentation with tutorials can be found here: https://raster4ml.readthedocs.io/en/latest/

## How to Use?
1. Stacking bands
    ```
    stack_bands(image_paths=['Band_1.tif', 'Band_2.tif', 'Band_3.tif',
                             'Band_4.tif', 'Band_5.tif', 'Band_6.tif'],
                out_file='Stack.tif')
    ```
2. Vegetation index calculation
    ```
    VI = VegetationIndices(image_path='Landsat8.tif',
                           wavelengths=[442.96, 482.04, 561.41, 654.59, 864.67, 1608.86, 2200.73])
    VI.calculate(out_dir='vegetation_indices')
    ```
2. Dynamic visualization in Jupyter Notebook
    ```
    m = Map()
    m.add_raster(image_path='Landsat8.tif', bands=[4, 3, 2])
    ```
    Output:
    ![map-output](https://raw.githubusercontent.com/souravbhadra/raster4ml/master/docs/images/map_output.png)

## How to Install?
### Dependencies
**Raster4ML** is built on top of [geopandas](https://geopandas.org/en/stable/), [rasterio](https://rasterio.readthedocs.io/en/latest/), [fiona](https://github.com/Toblerity/Fiona), [pyproj](https://pyproj4.github.io/pyproj/stable/), [rtree](https://github.com/Toblerity/rtree), [shapely](https://shapely.readthedocs.io/en/stable/manual.html), [numpy](https://numpy.org/), and [pandas](https://pandas.pydata.org/).

### Virtual Environment
It is prefered to use a virtual environment for working with this package. Use [Anaconda](https://www.anaconda.com/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to create a seperate environment and then install the package and its dependencies there.
```
conda create --name raster4ml python=3
conda activate raster4ml
```

### Windows
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

## Tutorials
There are two tutorials provided. Find them in ``docs/tutorials``.

## Questions?
Please report bugs at [https://github.com/remotesensinglab/raster4ml/issues](https://github.com/remotesensinglab/raster4ml/issues).

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.