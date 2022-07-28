======================================================================
Raster4ML: A geospatial raster processing library for machine learning
======================================================================

When geospatial raster data is concerned in a machine learning pipeline,
it is often required to extract meaningful features, such as vegetation
indices (e.g., NDVI, EVI, NDRE, etc.) or textures. This package provides
easy-to-use functions that can automatically calculates the features
with few lines of codes in Python. It also has the
functionality of extracting statistics based on shapefile (i.e., point
or polygon) from a raster data or multiple raster data at once. Any type of raster data is supported
regardless of satellite or UAVs.

Here is an example of creating as many vegetation index as the image wavelenghts support by
simply using following function:

.. code:: python
    from raster4ml.features import VegetaionIndices

    vi = VegetaionIndices(image_path='example.tif',
                          wavelengths=[[430, 450], [450, 510], [530, 590], [640, 670]])
    vi.calculate(out_path='/out_dir')


.. toctree::
   :maxdepth: 2

   installation
   quickstart
   raster4ml



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
