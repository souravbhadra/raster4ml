======================================================================
Raster4ML: A geospatial raster processing library for machine learning
======================================================================

Raster4ML is a python package that extracts machine learning ready dataset 
from geospatial raster data and shapefiles. The package aims to aid geospatial 
researchers and scientists to extract meaningful faetures easily and focus more 
on the model training or reproducibility issues.

Key Features
============
* Stack raster bands
* Automatically calculate vegetation indices (supports 350+ indices)
* Extract raster values based on shapefile
* Clip raster based on polygon

Raster4ML works with raster data derived from satellites, airplanes or UAVs. 
The applications can be supported in precision agriculture, plant phenotyping,
hydrology, natural resource management, and other fields within the geospatial 
science.

Here is an example of creating as many vegetation index as the image wavelenghts 
support by simply using the following function:

.. code:: python

    from raster4ml.features import VegetaionIndices

    VI = VegetationIndices(image_path='Landsat8.tif',
                           wavelengths=[442.96, 482.04, 561.41, 654.59, 864.67, 1608.86, 2200.73])
    VI.calculate(out_dir='vegetation_indices')



.. toctree::
   :maxdepth: 2

   intro
   installation
   quickstart
   topics/index
   Raster4ML API Reference <raster4ml>
   contributing

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
