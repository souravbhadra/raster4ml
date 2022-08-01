Introduction
============

Main Idea
---------

Machine learning has been playing a key role in different domains of geospatial science, 
such as, natural resource management, hydrology, agricultural monitoring, land cover dynamics, 
and so on. Researchers and scientists often use raster data derived from satellites, airplanes 
or unmanned aerial vehicles (UAVs) coupled with novel machine learning algorithms to estimate 
physiochemical parameters or explain the underlying processes of different phenomenon. 
Geospatial raster data is different from natural images often seen in computer vision 
applications. For example, a common task in utilizing machine learning for raster data is 
to derive hand-crafted features based on different disciplines or research questions. Such 
features have can explain certain characteristics, which cannot be interpreted by the 
individual bands or channels. To date, there has been many vegetation indices or texture 
features reported in literature. Therefore, it is difficult for researchers or scientists 
to derive the necessary feature from raster data and extract the values for the sample 
areas of interest. We hereby propose a Python-package called “Raster4ML”, which helps the 
users to easily create machine learning ready dataset from given geospatial data. The 
package can automatically calculate more than 350 vegetation indices and numerous texture 
features. It can also derive the statistical outputs from areas of interests for multiple 
raster data, which reduces the manual processing time for users. On the other hand, the 
ackage provides supports for dynamic visualization of the geospatial data that can help 
the users to interact with the inputs and outputs. The package will assist the geospatial 
community to derive meaningful features from geospatial datasets efficiently and automatically. 
This will enable them to focus more on the algorithm development, training, and 
reproducibility of the machine learning application rather than the preprocessing steps.

License
----------------

Copyright (c) 2022, Remote Sensing Lab, Sourav Bhadra, Vasit Sagan
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of Mapbox nor the names of its contributors may
  be used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.