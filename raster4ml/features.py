import os
from tqdm import tqdm
import numpy as np
import rasterio
from . import utils

class VegetationIndices():
    
    """Calculate vegetation indices from a given multispectral image.

    Attributes
    ----------
    image_path : str
        Path of the image.
    wavelengths : list or list of lists
        The wavelengths (in nanometer) of the bands in the stacked
        image. It can be either a list containing the center 
        wavelengths of the bands, or it can be a list of lists 
        where each element represents the initial and final wavelength.

        Example:
        If list: [430, 450, 560]
        If list of lists: [[420, 440], [530, 560], [670, 690]]
    bit_depth : int (optional)
        Bit depth of the image. For example, Landsat 8 images are
        14-bit. So if 14 is given as the bit_depth, then the image
        values will be divided by 2**14=16384.

    Methods
    -------
    calculate(out_dir, featuers='all'):
        Calculate all the vegetation indices possible.
        Allowed list of features are:
        'ARI_1', 'ARI_2', 'ARVI', 'CRI_1', 'CRI_2', 'DVI', 'EVI',
        'GEMI', 'GARI', 'GCI', 'GDVI', 'GLI', 'GNDVI', 'GOSAVI',
        'GRVI', 'GSAVI', 'GVI', 'IPVI', 'LCAI', 'MCARI', 'MNLI',
        'MNDWI', 'MRENDVI', 'MRESR', 'MSR', 'MSAVI_2', 'MTVI', 'MSI',
        'NLI', 'NBR', 'NBRT_1', 'NDBI', 'NDII', 'NDLI', 'NDMI', 
        'NDNI', 'NDSI', 'NDVI', 'NDWI', 'NMDI', 'OSAVI', 'PRI',
        'PSRI', 'RENDVI', 'RDVI', 'SR_1', 'SAVI', 'SIPI', 'TCARI',
        'TDVI', 'TVI', 'VARI', 'VREI_1', 'VREI_2', 'WBI', 'WDRVI'

    Raises
    ------
    ValueError
        If invalid image path is given.
    """
    
    def __init__(self, image_path, wavelengths, bit_depth=None):
        """Constructs all the necessary attributes for the 
        VegetationIndices object.

        Parameters
        ----------
        image_path : str
            Path of the image.
        wavelengths : list or list of lists
            The wavelengths (in nanometer) of the bands in the stacked
            image. It can be either a list containing the center 
            wavelengths of the bands, or it can be a list of lists 
            where each element represents the initial and final
            wavelength.

            Example:
            If list: [430, 450, 560]
            If list of lists: [[420, 440], [530, 560], [670, 690]]
        bit_depth : int (optional)
            Bit depth of the image. For example, Landsat 8 images are
            14-bit. So if 14 is given as the bit_depth, then the image
            values will be divided by 2**14=16384.

        Methods
        -------
        calculate(out_dir, featuers='all'):
            Calculate all the vegetation indices possible.
            Allowed list of features are:
            'ARI_1', 'ARI_2', 'ARVI', 'CRI_1', 'CRI_2', 'DVI', 'EVI',
            'GEMI', 'GARI', 'GCI', 'GDVI', 'GLI', 'GNDVI', 'GOSAVI',
            'GRVI', 'GSAVI', 'GVI', 'IPVI', 'LCAI', 'MCARI', 'MNLI',
            'MNDWI', 'MRENDVI', 'MRESR', 'MSR', 'MSAVI_2', 'MTVI',
            'MSI', 'NLI', 'NBR', 'NBRT_1', 'NDBI', 'NDII', 'NDLI',
            'NDMI', 'NDNI', 'NDSI', 'NDVI', 'NDWI', 'NMDI', 'OSAVI',
            'PRI', 'PSRI', 'RENDVI', 'RDVI', 'SR_1', 'SAVI', 'SIPI',
            'TCARI', 'TDVI', 'TVI', 'VARI', 'VREI_1', 'VREI_2', 'WBI',
            'WDRVI'

        Raises
        ------
        ValueError
            If invalid image path is given.
        """
        self.image_path = image_path
        self.src = rasterio.open(self.image_path)
        self.img = self.src.read()
        self.img = self.img.astype('float')
        self.img = np.moveaxis(self.img, 0, 2)
        
        # Define nodata
        if self.src.nodata is None:
            self.img[self.img==0] = np.nan
        else:
            self.img[self.img==self.src.nodata] = np.nan
        
        # Apply bit_depth if given
        if bit_depth is not None:
            if isinstance(bit_depth, int):
                self.img = self.img / 2.**bit_depth
            else:
                raise ValueError("Invalid input in bit_depth. Only int is accepted.")
            
        # Check if number of elements in wavelengths matches the number of bands
        if len(wavelengths) != self.img.shape[2]:
            raise ValueError("The number of elements in wavelengths and number of bands do not match.")
        
        # Find band wavelengths
        try:
            if any(isinstance(el, list) for el in wavelengths):
                self.wavelengths = [(el[0] + el[1])/2 for el in wavelengths]
            else: #list of lists
                self.wavelengths = [float(i) for i in wavelengths]
        except:
            raise ValueError("Invalid input of wavelengths. It has to \
                be a list of integers for center wavelengths and list \
                of lists with start and end wavelength for each band.")
            
        # Define the nearest bands
        self.R445 = self.nearest_band(445)
        self.R450 = self.nearest_band(450)
        self.R475 = self.nearest_band(475)
        self.R485 = self.nearest_band(485)
        self.R500 = self.nearest_band(500)
        self.R510 = self.nearest_band(510)
        self.R531 = self.nearest_band(531)
        self.R550 = self.nearest_band(550)
        self.R560 = self.nearest_band(560)
        self.R570 = self.nearest_band(570)
        self.R660 = self.nearest_band(660)
        self.R670 = self.nearest_band(670)
        self.R680 = self.nearest_band(680)
        self.R700 = self.nearest_band(700)
        self.R705 = self.nearest_band(705)
        self.R715 = self.nearest_band(715)
        self.R720 = self.nearest_band(720)
        self.R726 = self.nearest_band(726)
        self.R734 = self.nearest_band(734)
        self.R740 = self.nearest_band(740)
        self.R747 = self.nearest_band(747)
        self.R750 = self.nearest_band(750)
        self.R795 = self.nearest_band(795)
        self.R800 = self.nearest_band(800)
        self.R819 = self.nearest_band(819)
        self.R830 = self.nearest_band(830)
        self.R850 = self.nearest_band(850)
        self.R857 = self.nearest_band(857)
        self.R860 = self.nearest_band(860)
        self.R900 = self.nearest_band(900)
        self.R970 = self.nearest_band(970)
        self.R990 = self.nearest_band(990)
        self.R1145 = self.nearest_band(1145)
        self.R1241 = self.nearest_band(1241)
        self.R1510 = self.nearest_band(1510)
        self.R1599 = self.nearest_band(1599)
        self.R1640 = self.nearest_band(1640)
        self.R1649 = self.nearest_band(1649)
        self.R1650 = self.nearest_band(1650)
        self.R1680 = self.nearest_band(1680)
        self.R1754 = self.nearest_band(1754)
        self.R2000 = self.nearest_band(2000)
        self.R2100 = self.nearest_band(2100)
        self.R2130 = self.nearest_band(2130)
        self.R2165 = self.nearest_band(2165)
        self.R2200 = self.nearest_band(2200)
        self.R2205 = self.nearest_band(2205)
        self.R2215 = self.nearest_band(2215)
        self.R2330 = self.nearest_band(2330)
        
    
    def nearest_band(self, wavelength):
        """Find the nearest band of the image from given wavelength.

        Parameters
        ----------
        wavelength : float
            The wavelength in nm.

        Returns
        -------
        numpy nd-array
            The image numpy nd-array with selected bands matching the
            wavelength.
        """        
        difference = np.abs(np.array(self.wavelengths) - wavelength)
        if difference.min() < 100:
            return self.img[:, :, difference.argmin()]
        else:
            return None
        
    def ARI_1(self):
        """Anthocyanin Reflectance Index 1 (ARI1)
        URL: https://doi.org/10.1562/0031-8655(2001)074%3C0038:OPANEO%3E2.0.CO;2

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (1./self.R550)-(1./self.R700)

    def ARI_2(self):
        """Anthocyanin Reflectance Index 2 (ARI2)
        URL: https://doi.org/10.1562/0031-8655(2001)074%3C0038:OPANEO%3E2.0.CO;2

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R800*((1./self.R550)-(1./self.R700))

    def ARVI(self):
        """Atmospherically Resistant Vegetation Index (ARVI)
        URL: https://doi.org/10.1109/36.134076

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R800-(self.R680-(self.R450-self.R680)))/(self.R800+(self.R680-(self.R450-self.R680)))

    def CRI_1(self):
        """Carotenoid Reflectance Index 1 (CRI1)
        URL: https://calmit.unl.edu/people/agitelson2/pdf/08_2002_P&P_carotenoid.pdf

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (1./self.R510)-(1./self.R550)

    def CRI_2(self):
        """Carotenoid Reflectance Index 2 (CRI2)
        URL: https://calmit.unl.edu/people/agitelson2/pdf/08_2002_P&P_carotenoid.pdf

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (1./self.R510)-(1./self.R700)

    def CAI(self):
        """Cellulose Absorption Index (CAI)
        URL: https://naldc.nal.usda.gov/download/12951/PDF

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return 0.5*(self.R2000+self.R2200)-self.R2100

    def DVI(self):
        """Difference Vegetation Index (DVI)
        URL: https://doi.org/10.1016/0034-4257(79)90013-0

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R850-self.R660

    def EVI(self):
        """Enhanced Vegetation Index (EVI)
        URL: https://doi.org/10.1016/S0034-4257(02)00096-2

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return 2.5*((self.R850-self.R660)/(self.R850+(6.*self.R660)-(7.5*self.R475)+1.))

    def GEMI(self):
        """Global Environmental Monitoring Index (GEMI)
        URL: https://link.springer.com/article/10.1007/BF00031911

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        eta = (2.*(np.square(self.R850)-np.square(self.R660))+1.5*self.R850+0.5*self.R660)/(self.R850+self.R660+0.5)
        return eta*(1.-0.25*eta)-((self.R660-0.125)/(1.-self.R660))

    def GARI(self):
        """Green Atmospherically Resistant Index (GARI)
        URL: https://doi.org/10.1016/S0034-4257(96)00072-7

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R850-(self.R550-1.7*(self.R475-self.R660)))/(self.R850+(self.R550-1.7*(self.R475-self.R660)))

    def GCI(self):
        """Green Chlorophyll Index (GCI)
        URL: https://doi.org/10.1078/0176-1617-00887

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R850/self.R550)-1.

    def GDVI(self):
        """Green Difference Vegetation Index (GDVI)
        URL: https://repository.lib.ncsu.edu/handle/1840.16/4200

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R850-self.R550

    def GLI(self):
        """Green Leaf Index (GLI)
        URL: https://doi.org/10.1080/10106040108542184

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R550-self.R660+self.R550-self.R475)/(2.*self.R550+self.R660+self.R475)

    def GNDVI(self):
        """Green Normalized Difference Vegetation Index (GNDVI)
        URL: https://doi.org/10.1016/S0273-1177(97)01133-2

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """ 
        return (self.R850-self.R550)/(self.R850+self.R550)

    def GOSAVI(self):
        """Green Optimized Soil Adjusted Vegetation Index (GOSAVI)
        URL: https://repository.lib.ncsu.edu/handle/1840.16/4200

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return 'GOSAVI', (self.R850-self.R550)/(self.R850+self.R550+0.16)

    def GRVI(self):
        """Green Ratio Vegetation Index (GRVI)
        URL: https://doi.org/10.2134/agronj2005.0200

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """ 
        return self.R850/self.R550

    def GSAVI(self):
        """Green Soil Adjusted Vegetation Index (GSAVI)
        URL: https://repository.lib.ncsu.edu/handle/1840.16/4200

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return 1.5*((self.R850-self.R550)/(self.R850+self.R550+0.5))

    def GVI(self):
        """Green Vegetation Index (GVI)
        URL: https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.461.6381&rep=rep1&type=pdf

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (-0.2848*self.R485)+(-0.2435*self.R560)+(-0.5436*self.R660)+(0.7243*self.R830)+(0.0840*self.R1650)+(-0.1800*self.R2215)

    def IPVI(self):
        """Infrared Percentage Vegetation Index (IPVI)
        URL: https://doi.org/10.1016/0034-4257(90)90085-Z

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """ 
        return self.R850/(self.R850+self.R660)

    def LCAI(self):
        """Lignin Cellulose Absorption Index (LCAI)
        URL: https://doi.org/10.2134/agronj2003.0291

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return 100.*(self.R2205-self.R2165+self.R2205-self.R2330)

    def MCARI(self):
        """Modified Chlorophyll Absorption Ratio Index (MCARI)
        URL: https://doi.org/10.1016/S0034-4257(00)00113-9

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return ((self.R700-self.R670)-0.2*(self.R700-self.R550))*(self.R700/self.R670)

    def MNLI(self):
        """Modified Non-Linear Index (MNLI)
        URL: https://www.asprs.org/a/publications/proceedings/pecora17/0041.pdf

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """  
        return ((np.square(self.R850)-self.R660)*1.5)/(np.square(self.R850)+self.R660+0.5)

    def MNDWI(self):
        """Modified Normalized Difference Water Index (MNDWI)
        URL: https://doi.org/10.1080/01431160600589179
        URL: https://doi.org/10.1080/01431169608948714

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R550-self.R1650)/(self.R550+self.R1650)

    def MRENDVI(self):
        """Modified Red Edge Normalized Difference Vegetation Index (MRENDVI)
        URL: https://doi.org/10.1016/S0176-1617(99)80314-9
        URL: https://doi.org/10.1016/S0034-4257(02)00010-X

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """ 
        return (self.R750-self.R705)/(self.R750+self.R705-2.*self.R445)

    def MRESR(self):
        """Modified Red Edge Simple Ratio (MRESR)
        URL: https://doi.org/10.1016/S0034-4257(02)00010-X
        URL: https://doi.org/10.1016/S0176-1617(99)80314-9

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """ 
        return (self.R750-self.R445)/(self.R705-self.R445)

    def MSR(self):
        """Modified Red Edge Simple Ratio (MRESR)
        URL: https://doi.org/10.1080/07038992.1996.10855178

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """  
        return ((self.R850/self.R660)-1.)/(np.sqrt(self.R850/self.R660)+1.)

    def MSAVI_2(self):
        """Modified Soil Adjusted Vegetation Index 2 (MSAVI2)
        URL: https://doi.org/10.1016/0034-4257(94)90134-1

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """ 
        return (2.*self.R850+1.-np.sqrt(np.square(2.*self.R850)-8.*(self.R850-self.R660)))/2.

    def MTVI(self):
        """Modified Triangular Vegetation Index (MTVI)
        URL: https://doi.org/10.1016/j.rse.2003.12.013

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return 1.2*(1.2*(self.R800-self.R550)-2.5*(self.R670-self.R550))

    def MSI(self):
        """Moisture Stress Index (MSI)
        URL: https://doi.org/10.1016/0034-4257(89)90046-1
        URL: https://doi.org/10.1016/S0034-4257(01)00191-2

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R1599/self.R819

    def NLI(self):
        """Non-Linear Index (NLI)
        URL: https://doi.org/10.1080/02757259409532252

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (np.square(self.R850)-self.R660)/(np.square(self.R850)+self.R660)

    def NBR(self):
        """Normalized Burn Ratio (NBR)
        URL: https://doi.org/10.1080/10106049109354290
        URL: https://pubs.er.usgs.gov/publication/2002085

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R850-self.R2215)/(self.R850+self.R2215)

    def NBRT_1(self):
        """Normalized Burn Ratio Thermal 1 (NBRT1)
        URL: https://www.fs.usda.gov/treesearch/pubs/24608

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """ 
        return (self.R850-self.R2215*(self.R1145/1000.))/(self.R850+self.R2215*(self.R1145/1000.))

    def NDBI(self):
        """Normalized Difference Built-Up Index (NDBI)
        URL: https://doi.org/10.1080/01431160304987

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R1650-self.R830)/(self.R1650+self.R830)

    def NDII(self):
        """Normalized Difference Infrared Index (NDII)
        URL: https://www.asprs.org/wp-content/uploads/pers/1983journal/jan/1983_jan_77-83.pdf

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """ 
        return (self.R819-self.R1649)/(self.R819+self.R1649)

    def NDLI(self):
        """Normalized Difference Lignin Index (NDLI)
        URL: https://doi.org/10.1016/S0034-4257(02)00011-1
        URL: https://doi.org/10.1016/0034-4257(95)00234-0
        URL: https://doi.org/10.2307/1936780

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """ 
        return (np.log(1./self.R1754)-np.log(1./self.R1680))/(np.log(1./self.R1754)+np.log(1./self.R1680))

    def NDMI(self):
        """Normalized Difference Mud Index (NDMI)
        URL: https://doi.org/10.1117/1.OE.51.11.111719

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """ 
        return (self.R795-self.R990)/(self.R795+self.R990)

    def NDNI(self):
        """Normalized Difference Nitrogen Index (NDNI)
        URL: https://doi.org/10.1016/S0034-4257(02)00011-1
        URL: https://doi.org/10.1016/0034-4257(95)00234-0

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """ 
        return (np.log(1./self.R1510)-np.log(1./self.R1680))/(np.log(1./self.R1510)+np.log(1./self.R1680))

    def NDSI(self):
        """Normalized Difference Snow Index (NDSI)
        URL: https://doi.org/10.1016/0034-4257(95)00137-P
        URL: https://doi.org/10.1016/j.rse.2003.10.016

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R550-self.R1650)/(self.R550+self.R1650)

    def NDVI(self):
        """Normalized Difference Vegetation Index (NDVI)
        URL: https://ntrs.nasa.gov/citations/19740022614

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R850-self.R660)/(self.R850+self.R660)

    def NDWI(self):
        """Normalized Difference Water Index (NDWI)
        URL: https://doi.org/10.1016/S0034-4257(96)00067-3
        URL: https://doi.org/10.1016/j.rse.2003.10.021

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R857-self.R1241)/(self.R857+self.R1241)

    def NMDI(self):
        """Normalized Multi-band Drought Index (NMDI)
        URL: https://doi.org/10.1029/2007GL031021
        URL: https://doi.org/10.1016/j.agrformet.2008.06.005

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """ 
        return (self.R860-(self.R1640-self.R2130))/(self.R860+(self.R1640-self.R2130))

    def OSAVI(self):
        """Optimized Soil Adjusted Vegetation Index (OSAVI)
        URL: https://doi.org/10.1016/0034-4257(95)00186-7

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R850-self.R660)/(self.R850+self.R660+0.16)

    def PRI(self):
        """Photochemical Reflectance Index (PRI)
        URL: https://doi.org/10.1111/j.1469-8137.1995.tb03064.x
        URL: https://link.springer.com/article/10.1007/s004420050337

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R531-self.R570)/(self.R531+self.R570)

    def PSRI(self):
        """Plant Senescence Reflectance Index (PSRI)
        URL: https://doi.org/10.1034/j.1399-3054.1999.106119.x

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R680-self.R500)/self.R750

    def RENDVI(self):
        """Red Edge Normalized Difference Vegetation Index (RENDVI)
        URL: https://doi.org/10.1016/S0176-1617(11)81633-0
        URL: https://doi.org/10.1016/S0034-4257(02)00010-X

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """ 
        return (self.R750-self.R705)/(self.R750+self.R705)

    def RDVI(self):
        """Renormalized Difference Vegetation Index (RDVI)
        URL: https://doi.org/10.1016/0034-4257(94)00114-3

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R850-self.R660)/(np.sqrt(self.R850+self.R660))

    def SR_1(self):
        """Simple Ratio (SR_1)
        URL: https://doi.org/10.2134/agronj1968.00021962006000060016x

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R850/self.R660

    def SAVI(self):
        """Simple Ratio (SR_1)
        URL: https://doi.org/10.2134/agronj1968.00021962006000060016x

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """ 
        return (1.5*(self.R850-self.R660))/(self.R850+self.R660+0.5)

    def SIPI(self):
        """Structure Insensitive Pigment Index (SIPI)
        URL: https://publons.com/publon/483937/

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R800-self.R445)/(self.R800-self.R680)

    def TCARI(self):
        """Transformed Chlorophyll Absorption Reflectance Index (TCARI)
        URL: https://doi.org/10.1016/j.rse.2003.12.013

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return 3.*((self.R700-self.R670)-0.2*(self.R700-self.R550)*(self.R700/self.R670))

    def TDVI(self):
        """Transformed Difference Vegetation Index (TDVI)
        URL: https://doi.org/10.1109/IGARSS.2002.1026867

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return 1.5*((self.R850-self.R660)/(np.sqrt(np.square(self.R850)+self.R660+0.5)))

    def TVI(self):
        """Triangular Vegetation Index (TVI)
        URL: https://doi.org/10.1016/S0034-4257(00)00197-8

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (120.*(self.R750-self.R550)-200.*(self.R670-self.R550))/2.

    def VARI(self):
        """Visible Atmospherically Resistant Index (VARI)
        URL: https://doi.org/10.1080/01431160110107806

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R550-self.R660)/(self.R550+self.R660-self.R475)

    def VREI_1(self):
        """Vogelmann Red Edge Index 1 (VREI1)
        URL: https://doi.org/10.1080/01431169308953986

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R740/self.R720

    def VREI_2(self):
        """Vogelmann Red Edge Index 2 (VREI2)
        URL: https://doi.org/10.1080/01431169308953986

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R734-self.R747)/(self.R715-self.R726)

    def WBI(self):
        """Water Band Index (WBI)
        URL: https://doi.org/10.1080/01431169308954010
        URL: https://www.researchgate.net/publication/260000104_Mapping_crop_water_stress_issues_of_scale_in_the_detection_of_plant_water_status_using_hyperspectral_indices

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R970/self.R900

    def WDRVI(self):
        """Wide Dynamic Range Vegetation Index (WDRVI)
        URL: https://doi.org/10.1078/0176-1617-01176
        URL: https://digitalcommons.unl.edu/cgi/viewcontent.cgi?article=1264&context=natrespapers

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (0.2*self.R850-self.R660)/(0.2*self.R850+self.R660) 
    
    
    def calculate(self, out_dir, features='all'):
        
        """_summary_
        
        Parameters
        ----------
        out_dir : str
            Path of the directory where indice(s) will be saved.
        features : str or list (optional)
            If 'all' then all the vegetation indices available in the
            package will be calculated. Or the user can put the names
            of the indices to calculate in a list. Allowable indices
            are:
            'ARI_1', 'ARI_2', 'ARVI', 'CRI_1', 'CRI_2', 'DVI', 'EVI',
            'GEMI', 'GARI', 'GCI', 'GDVI', 'GLI', 'GNDVI', 'GOSAVI',
            'GRVI', 'GSAVI', 'GVI', 'IPVI', 'LCAI', 'MCARI', 'MNLI',
            'MNDWI', 'MRENDVI', 'MRESR', 'MSR', 'MSAVI_2', 'MTVI',
            'MSI', 'NLI', 'NBR', 'NBRT_1', 'NDBI', 'NDII', 'NDLI',
            'NDMI', 'NDNI', 'NDSI', 'NDVI', 'NDWI', 'NMDI', 'OSAVI',
            'PRI', 'PSRI', 'RENDVI', 'RDVI', 'SR_1', 'SAVI', 'SIPI',
            'TCARI', 'TDVI', 'TVI', 'VARI', 'VREI_1', 'VREI_2', 'WBI',
            'WDRVI'

        Returns
        -------
        None
            Saves the output indices.

        Raises
        ------
        ValueError
            _description_
        """        
        
        if features == 'all':
            features = utils.get_vegetation_indices_names()
            print(f'Calculating all features')
        elif isinstance(features, list):
            print(f'Calculating {len(features)} features')
        else:
            raise ValueError("Invalid input in features. Only support list of featues or 'all'.")

        not_calculated_features = []

        for feature in tqdm(features):
            out_path = os.path.join(out_dir, feature+'.tif')
            method = getattr(self, feature)
            
            try:
                feature = method()
                feature = feature.reshape(1, feature.shape[0], feature.shape[1])
                utils.save_raster(self.src, feature, out_path,
                                  driver='GTiff', dtype='float32', nodata=0, count=1)
            except Exception as e:
                not_calculated_features.append(feature)
                print(e)
                pass
        if len(not_calculated_features) > 0:
            print(f'{len(not_calculated_features)} features could not be calculated.')
        
        return None