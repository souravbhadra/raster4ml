============
Quick Start
============

For a quickstart, let's see how to pre-process a Landsat-8 OLI multispectral imagery so a 
huge number of vegetation indices can be calculated and visualized.

Import the functions.

.. code:: python

    from raster4ml.preprocessing import stack_bands
    from raster4ml.features import VegetationIndices
    from raster4ml.plotting import Map

1. Stacking bands

    .. code:: python

        stack_bands(image_paths=['Band_1.tif', 'Band_2.tif', 'Band_3.tif',
                                 'Band_4.tif', 'Band_5.tif', 'Band_6.tif'],
                    out_file='Stack.tif')


2. Vegetation index calculation

    .. code:: python

        VI = VegetationIndices(image_path='Landsat8.tif',
                               wavelengths=[442.96, 482.04, 561.41, 654.59, 864.67, 1608.86, 2200.73])
        VI.calculate(out_dir='vegetation_indices')


2. Dynamic visualization in Jupyter Notebook

    .. code:: python

        m = Map()
        m.add_raster(image_path='Landsat8.tif', bands=[4, 3, 2])
    

    .. image:: https://raw.githubusercontent.com/souravbhadra/raster4ml/master/docs/images/map_output.png
        :width: 400
        :alt: Map output

