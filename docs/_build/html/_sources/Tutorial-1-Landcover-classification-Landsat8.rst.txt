Tutorial 1: Landcover Classification using Landsat 8
====================================================

.. code:: ipython3

    import sys  
    sys.path.insert(0, r'F:\raster4ml')

.. code:: ipython3

    import os
    import glob
    from raster4ml.preprocessing import stack_bands
    from raster4ml.plotting import Map
    from raster4ml.features import VegetationIndices
    from raster4ml.extraction import batch_extract_values_by_points

1. Stack the Bands
------------------

First we need to stack all the bands together and make a multispectral
image file. The mutispectral image will contain several channels/bands
representing reflectance information from different wavelengths. Since
the test dataset is downloaded from a Landsat 8 satellite, there are
total 11 bands. However, we will only use the first 7 bands as they can
accurately define most of the surface objects in terms of reflectance.

To stack the seperate bands into one image, we need to define the paths
of all the bands in chronological order (actually any order you want,
but remember the orders for future reference).

.. code:: ipython3

    # Filter all the files that ends with .TIF
    image_dir = r'F:\raster4ml\data\landsat\LC08_L1TP_137045_20210317_20210328_01_T1'
    
    # Empty list to hold the first 7 bands' paths
    bands_to_stack = []
    # Loop through 7 times
    for i in range(7):
        bands_to_stack.append(os.path.join(image_dir,
                                           f'LC08_L1TP_137045_20210317_20210328_01_T1_B{i+1}.TIF'))
    bands_to_stack

.. code:: ipython3

    # Use the stack_bands function from raster4ml to do the stacking
    stack_bands(image_paths=bands_to_stack,
                out_file=os.path.join(image_dir, 'Stack.tif'))

Let’s visualize the image usign the plotting functionality of raster4ml.

.. code:: ipython3

    # Define the map instance
    m = Map()

.. code:: ipython3

    # Add the raster to the map
    m.add_raster(image_path=os.path.join(image_dir, 'Stack.tif'), bands=[4, 3, 2])

.. code:: ipython3

    m

2. Calculate Vegetation Indices
-------------------------------

In next step, we need to calculate the vegetation indices from the
stacked image. We can do this using
``raster4ml.features.VegetationIndices`` object. You can provide a list
of vegetation index we need to calculate in the object, but the tool can
automatically calcualte all the possible vegetation index rasters.

To do this, we need to provide the path of the stacked image, the
corresponding wavelength values and an output directory to save all the
indices as rasters. Since this is a Landsat 8 OLI image, we know the
band wavelengths. The wavelengths can be inserted as either the
``center_wavelengths`` as list or the range of wavelengths per band in a
list of list. The wavelengths has to be specified in nanometers (nm).
The Landsat 8 OLI wavelengths can be seen
`here <https://www.usgs.gov/faqs/what-are-band-designations-landsat-satellites>`__.

\*Optionally we can provide the ``bit_depth`` as a parameter. Since we
know Landsat 8 data is a 12-bit data, we can provide this information to
normalize the image values from 0 to 1.

.. code:: ipython3

    # Define the VegetationIndices object
    VI = VegetationIndices(image_path=r'F:\raster4ml\data\landsat\LC08_L1TP_137045_20210317_20210328_01_T1\Stack.tif',
                           wavelengths=[[430, 450], [450, 510], [530, 590], [640, 670], [850, 880], [1570, 1650], [2110, 2290]],
                           bit_depth=12)

.. code:: ipython3

    # Run the process while providing the output directory
    VI.calculate(out_dir=r'F:\raster4ml\data\landsat\LC08_L1TP_137045_20210317_20210328_01_T1\VIs')

Visualize any one of the index to see if the result is ok or not.

.. code:: ipython3

    # Add the raster to the map
    m.add_raster(image_path=r'F:\raster4ml\data\landsat\LC08_L1TP_137045_20210317_20210328_01_T1\VIs\NDVI.tif')

.. code:: ipython3

    m

3. Extract Values based on Sample Points
----------------------------------------

Locate the sample point shapefile in the ``data/shapes`` folder. The
name of the shapefile is ``points.shp``. We need to extract the
vegetation index values underneath each point in the shapefile and store
those index values for Machine Learning training. The shapefile also
contains label information. For simplicity, it only has two distinct
classes, i.e., ``Vegetation`` and ``Water``.

For extraction by points, we can use the
``raster4ml.extraction.batch_extract_values_by_points`` function. This
will enable extraction of multiple raster data at once. The function
takes ``image_paths`` as a list, ``shape_path`` as a string, and a
``unique_id`` in the shapefile which uniquely represent each point. The
function returns a pandas dataframe.

.. code:: ipython3

    # Visualize the shapefile onto the map
    m.add_shape(shape_path=r'F:\raster4ml\data\landsat\LC08_L1TP_137045_20210317_20210328_01_T1\shapes\points.shp', layer_control=True)

.. code:: ipython3

    m

.. code:: ipython3

    # Find the paths of all the vegetation indices
    vi_paths = glob.glob(r'F:\raster4ml\data\landsat\LC08_L1TP_137045_20210317_20210328_01_T1\VIs\*.tif')
    vi_paths




.. parsed-literal::

    ['F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\ARI_1.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\ARI_2.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\ARVI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\CRI_1.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\CRI_2.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\DVI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\EVI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\GARI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\GCI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\GDVI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\GEMI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\GLI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\GNDVI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\GRVI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\GSAVI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\GVI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\IPVI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\MCARI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\MNDWI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\MNLI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\MRENDVI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\MRESR.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\MSAVI_2.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\MSI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\MSR.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\MTVI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\NBR.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\NDBI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\NDII.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\NDSI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\NDVI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\NLI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\NMDI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\OSAVI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\PRI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\PSRI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\RDVI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\RENDVI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\SAVI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\SIPI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\SR_1.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\TCARI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\TDVI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\TVI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\VARI.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\VREI_1.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\VREI_2.tif',
     'F:\\raster4ml\\data\\landsat\\LC08_L1TP_137045_20210317_20210328_01_T1\\VIs\\WDRVI.tif']



.. code:: ipython3

    # Batch extract values by points
    values = batch_extract_values_by_points(image_paths=vi_paths,
                                            shape_path=r'F:\raster4ml\data\landsat\LC08_L1TP_137045_20210317_20210328_01_T1\shapes\points.shp',
                                            unique_id='UID')


.. parsed-literal::

    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████| 48/48 [00:29<00:00,  1.62it/s]

.. parsed-literal::

    6 columns were removed from the dataframe as they had duplicated values.
    

.. parsed-literal::

    
    

.. code:: ipython3

    values




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>ARI_1</th>
          <th>ARI_2</th>
          <th>ARVI</th>
          <th>CRI_1</th>
          <th>CRI_2</th>
          <th>DVI</th>
          <th>EVI</th>
          <th>GARI</th>
          <th>GCI</th>
          <th>GDVI</th>
          <th>...</th>
          <th>PSRI</th>
          <th>RDVI</th>
          <th>SAVI</th>
          <th>SIPI</th>
          <th>SR_1</th>
          <th>TCARI</th>
          <th>TDVI</th>
          <th>TVI</th>
          <th>VARI</th>
          <th>WDRVI</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>-0.062945</td>
          <td>-0.108279</td>
          <td>0.234300</td>
          <td>-0.065647</td>
          <td>-0.128593</td>
          <td>-0.199707</td>
          <td>0.102306</td>
          <td>0.213949</td>
          <td>-0.212297</td>
          <td>-0.463623</td>
          <td>...</td>
          <td>-0.327823</td>
          <td>-0.104673</td>
          <td>-0.072355</td>
          <td>5.270171</td>
          <td>0.895982</td>
          <td>0.158350</td>
          <td>-0.129161</td>
          <td>10.556641</td>
          <td>0.169782</td>
          <td>-0.696070</td>
        </tr>
        <tr>
          <th>1</th>
          <td>-0.081763</td>
          <td>-0.139712</td>
          <td>0.224136</td>
          <td>-0.051108</td>
          <td>-0.132871</td>
          <td>-0.245361</td>
          <td>0.114394</td>
          <td>0.191005</td>
          <td>-0.265274</td>
          <td>-0.616943</td>
          <td>...</td>
          <td>-0.350700</td>
          <td>-0.128203</td>
          <td>-0.088411</td>
          <td>4.550249</td>
          <td>0.874438</td>
          <td>0.222949</td>
          <td>-0.158764</td>
          <td>14.863281</td>
          <td>0.226522</td>
          <td>-0.702291</td>
        </tr>
        <tr>
          <th>2</th>
          <td>-0.078562</td>
          <td>-0.133685</td>
          <td>0.156079</td>
          <td>-0.039564</td>
          <td>-0.118126</td>
          <td>-0.395264</td>
          <td>0.175768</td>
          <td>0.120093</td>
          <td>-0.322182</td>
          <td>-0.808838</td>
          <td>...</td>
          <td>-0.329258</td>
          <td>-0.202804</td>
          <td>-0.137928</td>
          <td>3.162446</td>
          <td>0.811503</td>
          <td>0.248145</td>
          <td>-0.252982</td>
          <td>16.542969</td>
          <td>0.227230</td>
          <td>-0.720725</td>
        </tr>
        <tr>
          <th>3</th>
          <td>0.000882</td>
          <td>0.003013</td>
          <td>0.162228</td>
          <td>-0.010190</td>
          <td>-0.009307</td>
          <td>0.729492</td>
          <td>-14.067797</td>
          <td>0.142689</td>
          <td>0.274649</td>
          <td>0.735840</td>
          <td>...</td>
          <td>-0.025636</td>
          <td>0.295349</td>
          <td>0.165779</td>
          <td>0.693106</td>
          <td>1.271636</td>
          <td>-0.003809</td>
          <td>0.283973</td>
          <td>-0.253906</td>
          <td>-0.002432</td>
          <td>-0.594480</td>
        </tr>
        <tr>
          <th>4</th>
          <td>-0.037863</td>
          <td>-0.067416</td>
          <td>-0.005319</td>
          <td>-0.026779</td>
          <td>-0.064642</td>
          <td>-0.605957</td>
          <td>0.372762</td>
          <td>-0.028112</td>
          <td>-0.321329</td>
          <td>-0.843018</td>
          <td>...</td>
          <td>-0.182404</td>
          <td>-0.296846</td>
          <td>-0.194758</td>
          <td>1.968574</td>
          <td>0.746087</td>
          <td>0.142236</td>
          <td>-0.369330</td>
          <td>9.482422</td>
          <td>0.108334</td>
          <td>-0.740315</td>
        </tr>
        <tr>
          <th>...</th>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
          <td>...</td>
        </tr>
        <tr>
          <th>95</th>
          <td>-0.060378</td>
          <td>-0.227981</td>
          <td>0.509099</td>
          <td>-0.041061</td>
          <td>-0.101440</td>
          <td>1.830078</td>
          <td>-2.642041</td>
          <td>0.461469</td>
          <td>0.712546</td>
          <td>1.571045</td>
          <td>...</td>
          <td>-0.245922</td>
          <td>0.765082</td>
          <td>0.441218</td>
          <td>0.607924</td>
          <td>1.940527</td>
          <td>0.155420</td>
          <td>0.671681</td>
          <td>10.361328</td>
          <td>0.150049</td>
          <td>-0.440813</td>
        </tr>
        <tr>
          <th>96</th>
          <td>-0.053707</td>
          <td>-0.094210</td>
          <td>0.223083</td>
          <td>-0.068999</td>
          <td>-0.122705</td>
          <td>-0.169678</td>
          <td>0.092413</td>
          <td>0.214349</td>
          <td>-0.182408</td>
          <td>-0.391357</td>
          <td>...</td>
          <td>-0.309010</td>
          <td>-0.088475</td>
          <td>-0.060919</td>
          <td>5.771223</td>
          <td>0.911802</td>
          <td>0.133008</td>
          <td>-0.108518</td>
          <td>8.867188</td>
          <td>0.142925</td>
          <td>-0.691532</td>
        </tr>
        <tr>
          <th>97</th>
          <td>-0.044730</td>
          <td>-0.080789</td>
          <td>0.011139</td>
          <td>-0.021725</td>
          <td>-0.066455</td>
          <td>-0.544189</td>
          <td>0.341630</td>
          <td>-0.021856</td>
          <td>-0.312326</td>
          <td>-0.820312</td>
          <td>...</td>
          <td>-0.185104</td>
          <td>-0.266923</td>
          <td>-0.175300</td>
          <td>2.073127</td>
          <td>0.768464</td>
          <td>0.165674</td>
          <td>-0.330165</td>
          <td>11.044922</td>
          <td>0.126003</td>
          <td>-0.733564</td>
        </tr>
        <tr>
          <th>98</th>
          <td>-0.015176</td>
          <td>-0.029507</td>
          <td>-0.061845</td>
          <td>-0.015265</td>
          <td>-0.030441</td>
          <td>-0.604248</td>
          <td>0.607601</td>
          <td>-0.080921</td>
          <td>-0.266599</td>
          <td>-0.706787</td>
          <td>...</td>
          <td>-0.084108</td>
          <td>-0.285070</td>
          <td>-0.181531</td>
          <td>1.575758</td>
          <td>0.762908</td>
          <td>0.061523</td>
          <td>-0.346838</td>
          <td>4.101562</td>
          <td>0.042080</td>
          <td>-0.735235</td>
        </tr>
        <tr>
          <th>99</th>
          <td>-0.010707</td>
          <td>-0.039700</td>
          <td>0.225351</td>
          <td>-0.006250</td>
          <td>-0.016957</td>
          <td>1.082520</td>
          <td>-18.352650</td>
          <td>0.195860</td>
          <td>0.372650</td>
          <td>1.006592</td>
          <td>...</td>
          <td>-0.046592</td>
          <td>0.430161</td>
          <td>0.237638</td>
          <td>0.740189</td>
          <td>1.412350</td>
          <td>0.045557</td>
          <td>0.395307</td>
          <td>3.037109</td>
          <td>0.029442</td>
          <td>-0.559491</td>
        </tr>
      </tbody>
    </table>
    <p>100 rows × 42 columns</p>
    </div>



4. Machine Learning Training
----------------------------

Now that we have our data ready, let’s build our machine learning model
pipelines. We will explore two machine learning models, i.e., Support
Vector Machine (SVM) and Random Forest (RF) classification here. Our
target variable can be found in the point shapefile as the ``Label``
column. The independent variables will be the vegetation index values
calculated in the last step.

We will utilize functionalities from ``scikit-learn`` to train the
models. ``scikit-learn`` has an automatic ``pipeline`` feature that
performs several tasks at once. Machine learning models also require
**hyperparameter tuning** to fine tune the model. ``scikit-learn`` has a
fetaure for automatically doing that as well using ``GridSearchCV``. We
will employ all these steps at once using the pipeline.

Therfore install the ``scikit-learn`` using either ``pip`` or ``conda``
in the same environment and import the following modules.

.. code:: ipython3

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    
    import geopandas as gpd
    import numpy as np

.. code:: ipython3

    # Read the shapefile to get the points shapefile
    # Note that the rows of this shapefile and the extracted values match
    shape = gpd.read_file(r"F:\raster4ml\data\landsat\LC08_L1TP_137045_20210317_20210328_01_T1\shapes\points.shp")

.. code:: ipython3

    shape.head()




.. raw:: html

    <div>
    <style scoped>
        .dataframe tbody tr th:only-of-type {
            vertical-align: middle;
        }
    
        .dataframe tbody tr th {
            vertical-align: top;
        }
    
        .dataframe thead th {
            text-align: right;
        }
    </style>
    <table border="1" class="dataframe">
      <thead>
        <tr style="text-align: right;">
          <th></th>
          <th>Label</th>
          <th>UID</th>
          <th>geometry</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <th>0</th>
          <td>Water</td>
          <td>0</td>
          <td>POINT (193223.422 2349711.302)</td>
        </tr>
        <tr>
          <th>1</th>
          <td>Water</td>
          <td>1</td>
          <td>POINT (162754.153 2379518.334)</td>
        </tr>
        <tr>
          <th>2</th>
          <td>Water</td>
          <td>2</td>
          <td>POINT (124137.222 2358381.247)</td>
        </tr>
        <tr>
          <th>3</th>
          <td>Vegetation</td>
          <td>3</td>
          <td>POINT (224283.022 2424062.863)</td>
        </tr>
        <tr>
          <th>4</th>
          <td>Water</td>
          <td>4</td>
          <td>POINT (136454.441 2394163.347)</td>
        </tr>
      </tbody>
    </table>
    </div>



First, we need to split the dataset into training and testing set.

.. code:: ipython3

    X_train, X_test, y_train, y_test = train_test_split(values, shape['Label'].values, test_size=0.3, random_state=42)
    print('X_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    print('y_train shape:', y_train.shape)
    print('y_test shape:', y_test.shape)


.. parsed-literal::

    X_train shape: (70, 42)
    X_test shape: (30, 42)
    y_train shape: (70,)
    y_test shape: (30,)
    

Then, we just need to define the ``Pipeline``, ``GridSearchCV`` and the
model to do the training.

.. code:: ipython3

    ## Support Vector Machine
    
    # Define pipeline
    pipe_svc = Pipeline(steps=[('scaler', MinMaxScaler()), # Scaling the data from 0 to 1
                               ('model', SVC())])
    
    # Define pipeline parameters
    # Note that we are only testing 2 hyperparameters, you can do even more or expand the search
    param_svc = {'model__gamma': [2**i for i in np.arange(-10, 7, 1, dtype='float')],
                 'model__C': [2**i for i in np.arange(-10, 7, 1, dtype='float')]}
    
    # Define grid
    grid_svc = GridSearchCV(estimator=pipe_svc,
                            param_grid=param_svc,
                            cv=5, # 5-fold cross validation
                            n_jobs=4) # Paralelly using 4 CPU cores
    grid_svc.fit(X_train, y_train)




.. raw:: html

    <style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5,
                 estimator=Pipeline(steps=[(&#x27;scaler&#x27;, MinMaxScaler()),
                                           (&#x27;model&#x27;, SVC())]),
                 n_jobs=4,
                 param_grid={&#x27;model__C&#x27;: [0.0009765625, 0.001953125, 0.00390625,
                                          0.0078125, 0.015625, 0.03125, 0.0625,
                                          0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0,
                                          16.0, 32.0, 64.0],
                             &#x27;model__gamma&#x27;: [0.0009765625, 0.001953125, 0.00390625,
                                              0.0078125, 0.015625, 0.03125, 0.0625,
                                              0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0,
                                              16.0, 32.0, 64.0]})</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=5,
                 estimator=Pipeline(steps=[(&#x27;scaler&#x27;, MinMaxScaler()),
                                           (&#x27;model&#x27;, SVC())]),
                 n_jobs=4,
                 param_grid={&#x27;model__C&#x27;: [0.0009765625, 0.001953125, 0.00390625,
                                          0.0078125, 0.015625, 0.03125, 0.0625,
                                          0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0,
                                          16.0, 32.0, 64.0],
                             &#x27;model__gamma&#x27;: [0.0009765625, 0.001953125, 0.00390625,
                                              0.0078125, 0.015625, 0.03125, 0.0625,
                                              0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0,
                                              16.0, 32.0, 64.0]})</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;scaler&#x27;, MinMaxScaler()), (&#x27;model&#x27;, SVC())])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label sk-toggleable__label-arrow">MinMaxScaler</label><div class="sk-toggleable__content"><pre>MinMaxScaler()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label sk-toggleable__label-arrow">SVC</label><div class="sk-toggleable__content"><pre>SVC()</pre></div></div></div></div></div></div></div></div></div></div></div></div>



.. code:: ipython3

    ## Random Forest Classifier
    
    # Define pipeline
    pipe_rfc = Pipeline(steps=[('scaler', MinMaxScaler()), # Scaling the data from 0 to 1
                               ('model', RandomForestClassifier())])
    
    # Define pipeline parameters
    # Note that we are only testing 2 hyperparameters, you can do even more or expand the search
    param_rfc = {'model__n_estimators': [2**i for i in range(5)],
                 'model__max_features': ['sqrt', 'log2']}
    
    # Define grid
    grid_rfc = GridSearchCV(estimator=pipe_rfc,
                            param_grid=param_rfc,
                            cv=5, # 5-fold cross validation
                            n_jobs=4) # Paralelly using 4 CPU cores
    grid_rfc.fit(X_train, y_train)


.. parsed-literal::

    C:\Users\sbhadra\.conda\envs\raster4ml_dev\lib\site-packages\sklearn\ensemble\_forest.py:427: FutureWarning: `max_features='auto'` has been deprecated in 1.1 and will be removed in 1.3. To keep the past behaviour, explicitly set `max_features='sqrt'` or remove this parameter as it is also the default value for RandomForestClassifiers and ExtraTreesClassifiers.
      warn(
    



.. raw:: html

    <style>#sk-container-id-3 {color: black;background-color: white;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(cv=5,
                 estimator=Pipeline(steps=[(&#x27;scaler&#x27;, MinMaxScaler()),
                                           (&#x27;model&#x27;, RandomForestClassifier())]),
                 n_jobs=4,
                 param_grid={&#x27;model__max_features&#x27;: [&#x27;auto&#x27;, &#x27;sqrt&#x27;, &#x27;log2&#x27;],
                             &#x27;model__n_estimators&#x27;: [1, 2, 4, 8, 16]})</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" ><label for="sk-estimator-id-9" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(cv=5,
                 estimator=Pipeline(steps=[(&#x27;scaler&#x27;, MinMaxScaler()),
                                           (&#x27;model&#x27;, RandomForestClassifier())]),
                 n_jobs=4,
                 param_grid={&#x27;model__max_features&#x27;: [&#x27;auto&#x27;, &#x27;sqrt&#x27;, &#x27;log2&#x27;],
                             &#x27;model__n_estimators&#x27;: [1, 2, 4, 8, 16]})</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-10" type="checkbox" ><label for="sk-estimator-id-10" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: Pipeline</label><div class="sk-toggleable__content"><pre>Pipeline(steps=[(&#x27;scaler&#x27;, MinMaxScaler()),
                    (&#x27;model&#x27;, RandomForestClassifier())])</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-11" type="checkbox" ><label for="sk-estimator-id-11" class="sk-toggleable__label sk-toggleable__label-arrow">MinMaxScaler</label><div class="sk-toggleable__content"><pre>MinMaxScaler()</pre></div></div></div><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-12" type="checkbox" ><label for="sk-estimator-id-12" class="sk-toggleable__label sk-toggleable__label-arrow">RandomForestClassifier</label><div class="sk-toggleable__content"><pre>RandomForestClassifier()</pre></div></div></div></div></div></div></div></div></div></div></div></div>



Now that we have trained two models, lets check the accuracy score from
both models. We can directly use the ``grid`` objects. If we directly
predict from the ``grid`` object, then it picks out the model with the
best hyperparameters and use that for prediction. You can also go into
the ``grid`` object and examine which model to pick and so on. Please
refer
`here <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html>`__
to learn more.

.. code:: ipython3

    # Predict the test set
    y_pred_svc = grid_svc.predict(X_test)
    y_pred_rfc = grid_rfc.predict(X_test)

.. code:: ipython3

    print(f"Accuracy from SVC: {accuracy_score(y_test, y_pred_svc):.2f}")
    print(f"Accuracy from RFC: {accuracy_score(y_test, y_pred_rfc):.2f}")


.. parsed-literal::

    Accuracy from SVC: 0.97
    Accuracy from RFC: 1.00
    

