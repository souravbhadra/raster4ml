=====================================================================
Tutorial 2: Chlorophyll Estimation from UAV-borne Multispectral Image
=====================================================================

Preface
-------
The purpose of this tutorial is to estimate chlorophyll concentration of plants using 
machine learning. That's why a multispectral image with blue, green, red, red-edge, and
near infrared band is provided. A shapefile of some sample plots is also given. In the 
field, actual chlorophyll values of the plots were measured using destructive sampling. 
These can be used as ground truth for the machine learning model. However, to train a 
machine learning model, we need to extract remote sensing features or vegetation indices
under each plot. We can extract the average values of every vegetation indices for each 
plot and then train the model.

Datasets
--------
The dataset for this tutorial can be found in the ``raster4ml_data`` google drive repository_.
Download the ``chlorophyll.tar.gz`` file and extract in the data directory.


1. Micasense Altum Image
------------------------
The image provided was collected from a UAV-borne multispectral sensor called Micasense 
Altum. More information about Altum camera can be found here_. It has blue, green, red,
red-edge, and near infrared (NIR) bands. The channels in the attached image also follows 
the previously mentioned order of bands. The center wavelengths of the bandas can be found
in this link_.

2. Visualize
------------
Try to visualize the image using the ``plotting`` functionality of ``raster4ml``.


2. Calculate Vegetation Indices
-------------------------------
Calculate the vegetation indices from the ``micasense-altum-5bands.tif`` image. Please 
remember to consider the ``threshold`` parameter. Depending on the ``threshold`` you will 
get the desired amount of VIs. The image provided has a reflectance value ranging from 0 
to 1, which is why there is no need to provide any `bit_depth` information.

.. code:: python

    # Define the VegetationIndices object
    VI = VegetationIndices(`...`)

.. code:: python

    # Run the process while providing the output directory
    VI.calculate(out_dir='...')


3. Extract Values based on plot shape
----------------------------------------
Locate the sample polygon shapefile in the extracted data folder. The name of the shapefile 
is ``plot-shapefile.shp``. There are two columns in this shape, i.e., ``plotid`` and 
``chl``, where the first one is the unique id and the later is the measured chlorophyll
value. You can use the ``batch_extract_by_polygons`` from the ``extraction`` module. 
You can only extract the ``mean`` as the statistics.

.. code:: python

    # Batch extract values by polygons
    values = batch_extract_by_polygons(`...`)


4. Machine Learning Training
----------------------------
Now train at least 3 machine learning regression models of your choice. Remember to do 
feature scaling, feature selection, and hyperparameter optimization. Also, perform a 70/30
split of training and testing, where the models should be evaluated using the root mean 
squared error of the test set.


.. _repository: https://drive.google.com/drive/folders/1xzeYkVA-eYpeXJQ4Tu2OCDwtDBReAl6f?usp=sharing
.. _here: https://ageagle.com/drone-sensors/altum-pt/
.. _link: https://support.micasense.com/hc/en-us/articles/214878778-What-is-the-center-wavelength-and-bandwidth-of-each-filter-for-MicaSense-sensors-