=============
Installation
=============

Dependencies
------------
**Raster4ML** is built on top of geopandas_, rasterio_, fiona_, pyproj_, rtree_, shapely_, 
numpy_, and pandas_.

Virtual Environment
-------------------
It is prefered to use a virtual environment for working with this package. Use Anaconda_ 
or Miniconda_ to create a seperate environment and then install the package and its 
dependencies there.

.. code-block:: console

    conda create --name raster4ml python=3
    conda activate raster4ml

Windows
-------
To install on Windows, first download the wheel files for **GDAL**, **rasterio**, and 
**fiona** from Christoph Gohlke's website_ (🤗Thanks Christoph🤗). Go to his website, 
press ``Ctrl+F`` and type ``gdal``. Download the GDAL file that mostly matches your 
computer configuration (64-bit or 32-bit) and Python version.

After downloading it, ``cd`` into the downloaded directory while the ``raster4ml`` 
environment is activated. Then install using ``pip``. Do the same for **rasterio** and 
**fiona**.

.. code-block:: console

    pip install GDAL‑3.4.3‑cp310‑cp310‑win_amd64.whl
    pip install rasterio‑1.2.10‑cp310‑cp310‑win_amd64.whl
    pip install Fiona‑1.8.21‑cp310‑cp310‑win_amd64.whl


If these three are installed, the rest of the dependencies can be installed directly 
through **Raster4ML**'s ``pip`` distribution.

.. code-block:: console

    pip install raster4ml

Mac OS
------
Has not been tested yet. 😕

Linux
-----
Has not been tested yet. 😕



.. _geopandas: https://geopandas.org/en/stable/
.. _rasterio: https://rasterio.readthedocs.io/en/latest/
.. _fiona: https://github.com/Toblerity/Fiona
.. _pyproj: https://pyproj4.github.io/pyproj/stable/
.. _rtree: https://github.com/Toblerity/rtree
.. _shapely: https://shapely.readthedocs.io/en/stable/manual.html
.. _numpy: https://numpy.org/
.. _pandas: https://pandas.pydata.org/
.. _Anaconda: https://www.anaconda.com/
.. _Miniconda: https://docs.conda.io/en/latest/miniconda.html
.. _website: https://www.lfd.uci.edu/~gohlke/pythonlibs/

