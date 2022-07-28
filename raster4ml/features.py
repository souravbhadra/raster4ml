import os
from tqdm import tqdm
import numpy as np
import rasterio
from . import utils

import warnings
warnings.filterwarnings("ignore")

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
            

        
        
    
    def R(self, wavelength):
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
        if difference.min() < 10:
            return self.img[:, :, difference.argmin()]
        else:
            return None
        
    def ATSAVI(self):
        """Adjusted transformed soil-adjusted VI

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        x = 0.08
        a = 1.22
        b = 0.03
        return (a*(self.R(860) - a*self.R(650) - b))/(a*self.R(860) + self.R(650) - a*b + x*(1+a**2))

    def AFRI1600(self):
        """Aerosol free vegetation index 1600

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(860) - ((0.66*self.R(1600))/(self.R(860)+0.66*self.R(1600)))
    
    def AFRI2100(self):
        """Aerosol free vegetation index 2100

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(860) - ((0.5*self.R(2100))/(self.R(860)+0.56*self.R(2100)))
    
    def ARI(self):
        """	Anthocyanin reflectance index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (1/self.R(550))-(1/self.R(700))
    
    def AVI(self):
        """	Ashburn Vegetation Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return 2*self.R(860) - self.R(650)
    
    def ARVI(self):
        """	Atmospherically Resistant Vegetation Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(860)-self.R(650)-1*(self.R(650)-self.R(480))) / (self.R(860)+self.R(650)-1*(self.R(650)-self.R(480)))
    
    def ARVI2(self):
        """Atmospherically Resistant Vegetation Index 2

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return -0.18 + 1.17 * ((self.R(860)-self.R(650))/(self.R(860)+self.R(650)))
    
    def BWDRVI(self):
        """Blue-wide dynamic range vegetation index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (0.1*self.R(860)-self.R(480))/(0.1*self.R(860)+self.R(480))
    
    def BRI(self):
        """Browning Reflectance Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return ((1/self.R(550))-(1/self.R(700)))/self.R(860)
    
    def CCCI(self):
        """Canopy Chlorophyll Content Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return ((self.R(860)-self.R(715))/(self.R(860)+self.R(715))) / ((self.R(860)-self.R(650))/(self.R(860)+self.R(650)))
    
    def CASI_NDVI(self):
        """CASI NDVI

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return ((self.R(780)+self.R(790))-(self.R(665)+self.R(685))) / ((self.R(780)+self.R(790))+(self.R(665)+self.R(685)))
    
    def CASI_TM4_3(self):
        """CASI TM4/3

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(780)+self.R(790)) / (self.R(665)+self.R(685))
    
    def CARI(self):
        """Chlorophyll Absorption Ratio Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        a = (self.R(700)-self.R(550)) / 150
        b = (self.R(550)-((self.R(700)-self.R(550))/150*550))
        return (self.R(700)/self.R(670)) * (a*670+self.R(670)+b)/np.sqrt(a**2+1)
    
    def CARI2(self):
        """Chlorophyll Absorption Ratio Index 2

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        a = (self.R(700)-self.R(550))/150
        b = self.R(550)-(a*self.R(550))
        return (np.abs(a*self.R(670)+self.R(670)+b)/np.sqrt(a**2+1))*(self.R(700)/self.R(670))
    
    def Chlgreen(self):
        """Chlorophyll Green

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return np.power(self.R(800)/self.R(560), -1)
    
    def CIgreen(self):
        """Chlorophyll Index Green

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(800)/self.R(560))-1
    
    def CIrededge710(self):
        """Chlorophyll Index RedEdge 710

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(750)/self.R(710))-1
    
    def CIrededge(self):
        """Chlorophyll Index RedEdge 710

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(800)/self.R(715))-1
    
    def Chlrededge(self):
        """Chlorophyll RedEdge

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return np.power(self.R(800)/self.R(720), -1)
    
    def CVI(self):
        """Chlorophyll vegetation index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(860)*self.R(650))/np.square(self.R(560))
    
    def CI(self):
        """Coloration index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(650)-self.R(480))/self.R(650)
    
    def CTVI(self):
        """Corrected Transformed Vegetation Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return ((self.NDVI+0.5)/np.abs(self.NDVI+0.5))*np.sqrt(np.abs(self.NDVI+0.5))
    
    def CRI550(self):
        """CRI550

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return np.power(self.R(510), -1) - np.power(self.R(550), -1)
    
    def CRI700(self):
        """CRI700

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return np.power(self.R(510), -1) - np.power(self.R(700), -1)
    
    def CI(self):
        """Curvative index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(675)*(self.R(690)/np.square(self.R(683)))
    
    def Datt1(self):
        """Datt1

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(850)-self.R(710))/(self.R(850)-self.R(680))
    
    def Datt4(self):
        """Datt4

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(672)/(self.R(550)-self.R(708))
    
    def Datt6(self):
        """Datt6

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(860)/(self.R(550)-self.R(708))
    
    def D678_500(self):
        """Difference 678 & 500

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(678) - self.R(500)
    
    def D800_550(self):
        """Difference 800 & 550

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(800) - self.R(550)
    
    def D800_680(self):
        """Difference 800 & 680

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(800) - self.R(680)
    
    def D833_658(self):
        """Difference 833 & 658

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(678) - self.R(500)
    
    def GDVI(self):
        """Green Difference Vegetation Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(860) - self.R(560)
    
    def DVIMSS(self):
        """Differenced Vegetation Index MSS

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return 2.4*self.R(860) - self.R(650)
    
    def DSWI(self):
        """Disease water stress index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(802)+self.R(547))/(self.R(1657)+self.R(682))
    
    def DSWI5(self):
        """Disease water stress index 5

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(800)+self.R(550))/(self.R(1660)+self.R(680))
    
    def DD(self):
        """	Double Difference Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(749)-self.R(720))-(self.R(701)-self.R(672))
    
    def DPI(self):
        """Double Peak Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(688)+self.R(710))/np.square(self.R(697))
    
    def EVI(self):
        """Enhanced Vegetation Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return 2.5*((self.R(860)-self.R(650))/((self.R(860)+6*self.R(650)-7.5*self.R(480))+1))
    
    def EVI2(self):
        """Enhanced Vegetation Index 2

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return 2.4*((self.R(860)-self.R(650))/(self.R(860)+self.R(650)+1))
    
    def EVI3(self):
        """Enhanced Vegetation Index 3

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return 2.5*((self.R(860)-self.R(650))/(self.R(860)+2.4*self.R(650)+1))   
    
    def Gitelson(self):
        """Gitelson index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return ((self.R(750)-self.R(800))/(self.R(695)-self.R(740)))-1
    
    def GEMI(self):
        """Global Environment Monitoring Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        n = (2*(np.square(self.R(860))-np.square(self.R(650)))+1.5*self.R(860)+0.5*self.R(650)) / (self.R(860)+self.R(650)+0.5)
        return n*(1-0.25*n)-((self.R(650)-0.125)/(1-self.R(650)))
    
    def GVMI(self):
        """Global Vegetation Moisture Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return ((self.R(860)+0.1)-(self.R(2200)+0.02))/((self.R(860)+0.1)+(self.R(2200)+0.02))
    
    def GARI(self):
        """Green atmospherically resistant vegetation index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(860)-(self.R(560)-(self.R(480)-self.R(650)))) / (self.R(860)-(self.R(560)+(self.R(480)-self.R(650))))
    
    def GLI(self):
        """Green leaf index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (2*self.R(560)-self.R(650)-self.R(480))/(2*self.R(560)+self.R(650)+self.R(480))
    
    def GNDVI(self):
        """Green Normalized Difference Vegetation Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(860)-self.R(560))/(self.R(860)+self.R(560))
    
    def GOSAVI(self):
        """Green Optimized Soil Adjusted Vegetation Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(860)-self.R(560))/(self.R(860)+self.R(560)+0.16)
    
    def GSAVI(self):
        """Green Soil Adjusted Vegetation Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return ((self.R(860)-self.R(560))/(self.R(860)+self.R(560)+0.5))*1.5
    
    def GBNDVI(self):
        """Green-Blue NDVI

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(860)-(self.R(560)+self.R(480)))/(self.R(860)+(self.R(560)+self.R(480)))
    
    def Hue(self):
        """Hue

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return np.arctan2(((2*self.R(650)-self.R(560)-self.R(480))/30.5)*(self.R(560)-self.R(480)))
    
    def PVIhyp(self):
        """Hyperspectral perpendicular VI

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        a, b = 1.17, 3.37
        return (self.R(1148)-a*self.R(807)-b)/np.sqrt(1+a**2)
    
    def IPVI(self):
        """Infrared percentage vegetation index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return ((self.R(860)/(self.R(860)+self.R(650)))/2)*(self.NDVI+1)
    
    def Intensity(self):
        """Intensity

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (1/30.5)*(self.R(650)+self.R(560)+self.R(480))
    
    def IR550(self):
        """Inverse reflectance 550

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return np.power(self.R(550), -1)
    
    def IR700(self):
        """Inverse reflectance 700

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return np.power(self.R(700), -1)
    
    def LCI(self):
        """Leaf Chlorophyll Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(850)-self.R(710))/(self.R(850)+self.R(680))
    
    def LWCI(self):
        """Leaf Water Content Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return np.log(1-(self.R(860)-self.R(2215)))/-(np.log(1-(self.R(860)-self.R(2215))))
    
    def LogR(self):
        """Log ratio

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return np.log(self.R(860)/self.R(650))
    
    def Maccioni(self):
        """Maccioni

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(780)-self.R(710))/(self.R(780)-self.R(680))
    
    def mCRIG(self):
        """mCRIG

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (np.power(self.R(520), -1) - np.power(self.R(570), -1))*self.R(860)
    
    def mCRIRE(self):
        """mCRIG

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (np.power(self.R(520), -1) - np.power(self.R(700), -1))*self.R(860)
    
    def MTCI(self):
        """MERIS Terrestrial chlorophyll index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(754)-self.R(709))/(self.R(709)-self.R(681))
    
    def MVI(self):
        """Mid-infrared vegetation index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(700)-self.R(1570)
    
    def MGVI(self):
        """Misra Green Vegetation Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return -0.386*self.R(550) - 0.530*self.R(650) + 0.535*self.R(750) + 0.532*self.R(950)
    
    def MNSI(self):
        """Misra Non Such Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return 0.404*self.R(550) - 0.039*self.R(650) + 0.505*self.R(750) + 0.762*self.R(950)
    
    def MSBI(self):
        """Misra Soil Brightness Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return 0.406*self.R(550) + 0.600*self.R(650) + 0.645*self.R(750) + 0.243*self.R(950)
    
    def MYVI(self):
        """Misra Yellow Vegetation Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return 0.723*self.R(550) - 0.597*self.R(650) + 0.206*self.R(750) - 0.278*self.R(950)
    
    def mND680(self):
        """mND680

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(800)-self.R(680))/(self.R(800)+self.R(680)-self.R(2445))
    
    def mARI(self):
        """Modified anthocyanin reflectance index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (np.power(self.R(570), -1)-np.power(self.R(710), -1))*self.R(860)
    
    def MCARI(self):
        """Modified Chlorophyll Absorption in Reflectance Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return ((self.R(700)-self.R(670))-0.2*(self.R(700)-self.R(550)))*(self.R(700)/self.R(670))
    
    def MCARI1(self):
        """Modified Chlorophyll Absorption in Reflectance Index 1

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return 1.2*(2.5*(800-670)-1.3*(800-550))
    
    def MCARI1510(self):
        """Modified Chlorophyll Absorption in Reflectance Index 1510

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return ((self.R(700)-self.R(1510))-0.2*(self.R(700)-self.R(550)))*(self.R(700)/self.R(1510))
    
    def MCARI2(self):
        """Modified Chlorophyll Absorption in Reflectance Index 2

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return 1.5 * ((2.5*(self.R(800)-self.R(670))-1.3*(self.R(800)-self.R(550)))/np.sqrt(np.square(2*self.R(800)+1)-(6*self.R(800)-5*np.sqrt(self.R(670)))-0.5))
    
    def MCARI705(self):
        """Modified Chlorophyll Absorption in Reflectance Index 705, 750

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return ((self.R(750)-self.R(705))-0.2*(self.R(750)-self.R(550)))*(self.R(750)/self.R(705))
    
    def MCARI710(self):
        """Modified Chlorophyll Absorption in Reflectance Index 710, 750

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return ((self.R(750)-self.R(710))-0.2*(self.R(750)-self.R(550)))*(self.R(750)/self.R(710))
    
    def mNDVI(self):
        """Modified NDVI

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(800)-self.R(680))/(self.R(800)+self.R(680)-self.R(2445))
    
    def Vog2(self):
        """Modified Normalised Difference 734/747/715/726 Vogelmann indices 2

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(734)-self.R(747))/(self.R(715)+self.R(726))
    
    def MND_750_705(self):
        """Modified Normalised Difference 750/705

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(750)-self.R(705))/(self.R(750)+self.R(705)-self.R(2445))
    
    def MND_734_747_715_720(self):
        """Modified Normalized Difference 734/747/715/720

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(734)-self.R(747))/(self.R(715)-self.R(720))
    
    def MND_850_1788_1928(self):
        """Modified Normalized Difference 850/1788/1928

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(850)-self.R(1788))/(self.R(850)+self.R(1928))
    
    def MND_850_2218_1928(self):
        """Modified Normalized Difference 850/2218/1928

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(850)-self.R(2218))/(self.R(850)+self.R(1928))
    
    def MRVI(self):
        """Modified Normalized Difference Vegetation Index RVI

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.RVI-1)/(self.RVI+1)
    
    def mSR(self):
        """Modified Simple Ratio

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(800)-self.R(445))/(self.R(680)-self.R(445))
    
    def MSR670(self):
        """Modified Simple Ratio 670, 800

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return ((self.R(800)/self.R(670))-1)/np.sqrt((self.R(800)/self.R(670))+1)
    
    def MSR705(self):
        """Modified Simple Ratio 705, 750

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return ((self.R(750)/self.R(705))-1)/np.sqrt((self.R(750)/self.R(705))+1)
    
    def MSR_705_445(self):
        """Modified Simple Ratio 705, 445

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(750)-self.R(445))/(self.R(705)-self.R(445))
    
    def MSR_NIR_Red(self):
        """Modified Simple Ratio NIR Red

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return ((self.R(860)/self.R(650))-1)/np.sqrt((self.R(860)/self.R(650))+1)
    
    def MSAVI(self):
        """Modified Soil Adjusted Vegetation Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (2*self.R(860)+1-np.sqrt(np.square(self.R(860)+1)-8*(self.R(860)-self.R(650))))/2
    
    def MSAVIhyper(self):
        """Modified Soil Adjusted Vegetation Index hyper

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (2*self.R(800)+1-np.sqrt(np.square(self.R(800)+1)-8*(self.R(800)-self.R(670))))/2
    
    def MTVI1(self):
        """Modified Triangular Vegetation Index 1

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return 1.2*(1.2*(self.R(800)-self.R(550))-2.5*(self.R(670)-self.R(550)))
    
    def MTVI2(self):
        """Modified Triangular Vegetation Index 2

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return 1.5*((1.2*(self.R(800)-self.R(550))-2.5*(self.R(670)-self.R(550))) / np.sqrt(np.square(2*self.R(800)+1)-(6*self.R(800)-5*np.sqrt(self.R(670)))-0.5))
    
    def MNLI(self):
        """Modified NLI

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (1.5*(np.square(self.R(1760))-self.R(824)))/(np.square(self.R(1760))+self.R(824)+0.5)
    
    def mSR2(self):
        """mSR2

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(750)/self.R(705))-(1/np.sqrt((self.R(750)/self.R(705))+1))
    
    def DDn(self):
        """Double Difference Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return 2*(self.R(710)-self.R(760)-self.R(760))
    
    def NLI(self):
        """Nonlinear vegetation index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (np.square(self.R(860))-self.R(650))/(np.square(self.R(860))+self.R(650))
    
    def NormG(self):
        """Normalized Green

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(560)/(self.R(860)+self.R(650)+self.R(560))
    
    def NormNIR(self):
        """Normalized NIR

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(860)/(self.R(860)+self.R(650)+self.R(560))
    
    def NormR(self):
        """Normalized Red

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(650)/(self.R(860)+self.R(650)+self.R(560))
    
    def NDWI_Hyp(self):
        """Normalized difference water index hyperion

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(1070)-self.R(1200)) / (self.R(1070)+self.R(1200))
    
    def ND_1080_1180(self):
        """Normalized difference 1080 1180

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(1080)-self.R(1180)) / (self.R(1080)+self.R(1180))
    
    def ND_1080_1260(self):
        """Normalized difference 1080 1260

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(1080)-self.R(1260)) / (self.R(1080)+self.R(1260))
    
    def ND_1080_1450(self):
        """Normalized difference 1080 1450

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(1080)-self.R(1450)) / (self.R(1080)+self.R(1450))
    
    def ND_1080_1675(self):
        """Normalized difference 1080 1675

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(1080)-self.R(1675)) / (self.R(1080)+self.R(1675))
    
    def ND_1080_2170(self):
        """Normalized difference 1080 2170

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(1080)-self.R(2170)) / (self.R(1080)+self.R(2170))
    
    def LWVI1(self):
        """Leaf water vegetation index 1

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(1094)-self.R(893)) / (self.R(1094)+self.R(893))
    
    def LWVI2(self):
        """Leaf water vegetation index 1

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(1094)-self.R(1205)) / (self.R(1094)+self.R(1205))
    
    def ND_1180_1450(self):
        """Normalized difference 1180 1450

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(1180)-self.R(1450)) / (self.R(1180)+self.R(1450))
    
    def ND_1180_1675(self):
        """Normalized difference 1180 1675

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(1180)-self.R(1675)) / (self.R(1180)+self.R(1675))
    
    def ND_1180_2170(self):
        """Normalized difference 1180 2170

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(1180)-self.R(2170)) / (self.R(1180)+self.R(2170))
    
    def ND_1260_1450(self):
        """Normalized difference 1260 1450

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(1260)-self.R(1450)) / (self.R(1260)+self.R(1450))
    
    def ND_1260_1675(self):
        """Normalized difference 1260 1675

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(1260)-self.R(1675)) / (self.R(1260)+self.R(1675))
    
    def ND_1260_2170(self):
        """Normalized difference 1260 2170

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(1260)-self.R(2170)) / (self.R(1260)+self.R(2170))
    
    def ND_1510_660(self):
        """Normalized difference 1260 660

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(1510)-self.R(660)) / (self.R(1510)+self.R(660))
    
    def NDBleaf(self):
        """Normalized Difference leaf canopy biomass

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(2160)-self.R(1540)) / (self.R(2160)+self.R(1540))
    
    def NDlma(self):
        """Normalized Difference leaf mass per area

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(2260)-self.R(1490)) / (self.R(2260)+self.R(1490))
    
    def NPQI(self):
        """Normalized Phaeophytinization Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(415)-self.R(435)) / (self.R(415)+self.R(435))
    
    def PRI_528_587(self):
        """Photochemical Reflectance Index 528/587

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(528)-self.R(587)) / (self.R(528)+self.R(587))
    
    def PRI_531_570(self):
        """Photochemical Reflectance Index 531/570

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(531)-self.R(570)) / (self.R(531)+self.R(570))
    
    def PPR(self):
        """Plant pigment ratio

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(550)-self.R(450)) / (self.R(550)+self.R(450))
    
    def PRI_550_530(self):
        """Photochemical Reflectance Index 550/530

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(550)-self.R(530)) / (self.R(550)+self.R(530))
    
    def PVR(self):
        """Photosynthetic vigour ratio

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(550)-self.R(650)) / (self.R(550)+self.R(650))
    
    def PRI_570_539(self):
        """Photochemical Reflectance Index 570/539

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(570)-self.R(539)) / (self.R(570)+self.R(539))
    
    def PRI_570_531(self):
        """Photochemical Reflectance Index 570/531

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(570)-self.R(531)) / (self.R(570)+self.R(531))
    
    def NPCI(self):
        """Normalized Pigment Chlorophyll Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(680)-self.R(430)) / (self.R(680)+self.R(430))
    
    def ND_682_553(self):
        """Normalized difference 682 553

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(682)-self.R(553)) / (self.R(682)+self.R(553))
    
    def NDVIg(self):
        """Green NDVI

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(750)-self.R(550)) / (self.R(750)+self.R(550))
    
    def ND_750_650(self):
        """Normalized difference 750 650

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(750)-self.R(650)) / (self.R(750)+self.R(650))
    
    def ND_750_660(self):
        """Normalized difference 750 660

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(750)-self.R(660)) / (self.R(750)+self.R(660))
    
    def ND_750_680(self):
        """Normalized difference 750 680

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(750)-self.R(680)) / (self.R(750)+self.R(680))
    
    def NDVI_705(self):
        """Chl NDVI

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(750)-self.R(705)) / (self.R(750)+self.R(705))
    
    def RENDVI(self):
        """Red edge NDVI

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(750)-self.R(710)) / (self.R(750)+self.R(710))
    
    def ND_774_677(self):
        """Normalized difference 774 677

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(774)-self.R(677)) / (self.R(774)+self.R(677))
    
    def GNDVIhyper(self):
        """Green NDVI hyper

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(780)-self.R(550)) / (self.R(780)+self.R(550))
    
    def ND_782_666(self):
        """Normalized difference 782 666

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(782)-self.R(666)) / (self.R(782)+self.R(666))
    
    def ND_790_670(self):
        """Normalized difference 790 670

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(790)-self.R(670)) / (self.R(790)+self.R(670))
    
    def NDRE(self):
        """Normalized difference red edge index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(790)-self.R(720)) / (self.R(790)+self.R(720))
    
    def ND_800_1180(self):
        """Normalized difference 800 1180

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(800)-self.R(1180)) / (self.R(800)+self.R(1180))
    
    def ND_800_1260(self):
        """Normalized difference 800 1260

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(800)-self.R(1260)) / (self.R(800)+self.R(1260))
    
    def ND_800_1450(self):
        """Normalized difference 800 1450

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(800)-self.R(1450)) / (self.R(800)+self.R(1450))
    
    def ND_800_1675(self):
        """Normalized difference 800 1675

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(800)-self.R(1675)) / (self.R(800)+self.R(1675))
    
    def ND_800_2170(self):
        """Normalized difference 800 2170

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(800)-self.R(2170)) / (self.R(800)+self.R(2170))
    
    def PSNDc2(self):
        """Pigment specific normalised difference C2

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(800)-self.R(470)) / (self.R(800)+self.R(470))
    
    def PSNDc1(self):
        """Pigment specific normalised difference C1

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(800)-self.R(500)) / (self.R(800)+self.R(500))
    
    def GNDVIhyper2(self):
        """Green NDVI hyper 2

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(800)-self.R(550)) / (self.R(800)+self.R(550))
    
    def PSNDb2(self):
        """Pigment specific normalised difference B2

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(800)-self.R(635)) / (self.R(800)+self.R(635))
    
    def PSNDb1(self):
        """Pigment specific normalised difference B1

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(800)-self.R(650)) / (self.R(800)+self.R(650))
    
    def PSNDa1(self):
        """Pigment specific normalised difference A1

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(800)-self.R(675)) / (self.R(800)+self.R(675))
    
    def ND_800_680(self):
        """Normalized difference 800 680

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(800)-self.R(680)) / (self.R(800)+self.R(680))
    
    def ND_819_1600(self):
        """Normalized difference 819 1600

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(819)-self.R(1600)) / (self.R(819)+self.R(1600))
    
    def ND_819_1649(self):
        """Normalized difference 819 1649

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(819)-self.R(1649)) / (self.R(819)+self.R(1649))
    
    def NDMI(self):
        """Normalized Difference Moisture Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(820)-self.R(1600)) / (self.R(820)+self.R(1600))
    
    def ND_827_668(self):
        """Normalized difference 827 668

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(827)-self.R(668)) / (self.R(827)+self.R(668))
    
    def ND_833_1649(self):
        """Normalized difference 833 1649

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(833)-self.R(1649)) / (self.R(833)+self.R(1649))
    
    def ND_833_658(self):
        """Normalized difference 833 658

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(833)-self.R(658)) / (self.R(833)+self.R(658))
    
    def ND_850_1650(self):
        """Normalized difference 850 1650

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(850)-self.R(1650)) / (self.R(850)+self.R(1650))
    
    def ND_857_1241(self):
        """Normalized difference 857 1241

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(857)-self.R(1241)) / (self.R(857)+self.R(1241))
    
    def ND_860_1240(self):
        """Normalized difference 860 1240

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(860)-self.R(1240)) / (self.R(860)+self.R(1240))
    
    def ND_895_675(self):
        """Normalized difference 895 675

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(895)-self.R(675)) / (self.R(895)+self.R(675))
    
    def ND_900_680(self):
        """Normalized difference 900 680

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(900)-self.R(680)) / (self.R(895)+self.R(680))
    
    def NDchl(self):
        """Normalized Difference Chlorophyll

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(925)-self.R(710)) / (self.R(925)+self.R(710))
    
    def ND_960_1180(self):
        """Normalized difference 960 1180

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(960)-self.R(1180)) / (self.R(895)+self.R(1180))
    
    def ND_960_1260(self):
        """Normalized difference 960 1260

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(960)-self.R(1260)) / (self.R(895)+self.R(1260))
    
    def ND_960_1450(self):
        """Normalized difference 960 1450

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(960)-self.R(1450)) / (self.R(895)+self.R(1450))
    
    def ND_960_1675(self):
        """Normalized difference 960 1675

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(960)-self.R(1675)) / (self.R(895)+self.R(1675))
    
    def ND_960_2170(self):
        """Normalized difference 960 2170

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(960)-self.R(2170)) / (self.R(895)+self.R(2170))
    
    def NGRDI(self):
        """Normalized Green Red Difference Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(560)-self.R(650)) / (self.R(560)+self.R(650))
    
    def NDLI(self):
        """Normalized Difference Lignin Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (np.log(1/self.R(1754))-np.log(1/self.R(1680))) / (np.log(1/self.R(1754))+np.log(1/self.R(1680)))
    
    def NDVI(self):
        """Normalized Difference Vegetation Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(860)-self.R(650)) / (self.R(860)+self.R(650))
    
    def BNDVI(self):
        """Blue Normalized Difference Vegetation Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(860)-self.R(480)) / (self.R(860)+self.R(480))
    
    def GNDVI(self):
        """Blue Normalized Difference Vegetation Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(860)-self.R(560)) / (self.R(860)+self.R(560))
    
    def NDRE(self):
        """Normalized Difference Rededge Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(860)-self.R(715)) / (self.R(860)+self.R(715))
    
    def NBR(self):
        """Normalized Burn Ratio

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(860)-self.R(2200)) / (self.R(860)+self.R(2200))
    
    def NDNI(self):
        """Normalized Difference Nitrogen Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (np.log(1/self.R(1510))-np.log(1/self.R(1680))) / (np.log(1/self.R(1510))+np.log(1/self.R(1680)))
    
    def RI(self):
        """Normalized Burn Ratio

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(650)-self.R(560)) / (self.R(650)+self.R(560))
    
    def NDVI_rededge(self):
        """Normalized Difference Rededge/Red

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(715)-self.R(650)) / (self.R(715)+self.R(650))
    
    def NDSI(self):
        """Normalized Difference Salinity Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(1650)-self.R(2160)) / (self.R(1650)+self.R(2160))
    
    def NDVI_700(self):
        """NDVI 700

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(860)-self.R(700)) / (self.R(860)+self.R(700))
    
    def OSAVI(self):
        """Optimized Soil Adjusted Vegetation Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return 1.16*((self.R(800)-self.R(670))/(self.R(800)+self.R(670)+0.16))
    
    def OSAVI_1510(self):
        """Optimized Soil Adjusted Vegetation Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return 1.16*((self.R(800)-self.R(1510))/(self.R(800)+self.R(1510)+0.16))
    
    def OSAVI2(self):
        """Optimized Soil Adjusted Vegetation Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return 1.16*((self.R(750)-self.R(705))/(self.R(750)+self.R(705)+0.16))
    
    def PNDVI(self):
        """Pan NDVI

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(860)-(self.R(560)+self.R(650)+self.R(480))) / (self.R(860)+(self.R(560)+self.R(650)+self.R(480)))
    
    def PSRI(self):
        """Plant Senescence Reflectance Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(678)-self.R(500)) / self.R(750)
    
    def R_675_700_650(self):
        """Ratio 675 700 650

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(678) / (self.R(700) * self.R(650))
    
    def R_WI_ND750(self):
        """Ratio 675 700 650

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(900)/self.R(970)) / ((self.R(750)-self.R(705))/(self.R(750)+self.R(705)))
    
    def RDVI(self):
        """RDVI

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(800)-self.R(670)) / np.sqrt(self.R(800)+self.R(670))
    
    def RDVI2(self):
        """RDVI

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(833)-self.R(658)) / np.sqrt(self.R(833)+self.R(658))
    
    def Rededge1(self):
        """Rededge1

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(716) / self.R(685)
    
    def Rededge2(self):
        """Rededge1

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(716)-self.R(685)) / (self.R(716)+self.R(685))
    
    def RBNDVI(self):
        """Red-Blue NDVI

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(860)-(self.R(650)+self.R(480))) / (self.R(860)+(self.R(650)+self.R(480)))
    
    def REIP1(self):
        """Red-Edge Inflection Point 1

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return 700+40*(((0.5*(self.R(670)+self.R(780)))-self.R(700))/(self.R(740)-self.R(700)))
    
    def REIP2(self):
        """Red-Edge Inflection Point 2

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return 702+40*(((0.5*(self.R(667)+self.R(782)))-self.R(702))/(self.R(742)-self.R(702)))
    
    def REIP3(self):
        """Red-Edge Inflection Point 3

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return 705+35*(((0.5*(self.R(665)+self.R(783)))-self.R(705))/(self.R(740)-self.R(705)))
    
    def REP(self):
        """Red-Edge Position Linear Interpolation

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return 705+40*(((0.5*(self.R(670)+self.R(780)))-self.R(700))/(self.R(740)-self.R(700)))
    
    def RVSI(self):
        """Red-Edge Stress Vegetation Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return 0.5*(self.R(718)+self.R(748)) - self.R(733)
    
    def RRE(self):
        """Reflectance at the inflexion point

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return 0.5*(self.R(670)+self.R(780))
    
    def RDVI(self):
        """Renormalized Difference Vegetation Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(800)-self.R(670)) / np.sqrt(self.R(800)+self.R(670))
    
    def SAVImir(self):
        """SAVImir

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return (self.R(860)-self.R(2200)) * (1.16/(self.R(860)+self.R(2200)+0.16))
    
    def IF(self):
        """Shape Index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return 2*650-560-480
    
    def SR_1058_1148(self):
        """Simple Ratio of 1058 1148

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(1058)/self.R(1148)
    
    def SR_1080_1180(self):
        """Simple Ratio of 1080 1180

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(1080)/self.R(1180)
    
    def SR_1080_1260(self):
        """Simple Ratio of 1080 1260

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(1080)/self.R(1260)
    
    def SR_1080_1450(self):
        """Simple Ratio of 1080 1450

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(1080)/self.R(1450)
    
    def SR_1080_1675(self):
        """Simple Ratio of 1080 1675

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(1080)/self.R(1675)
    
    def SR_1080_2170(self):
        """Simple Ratio of 1080 2170

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(1080)/self.R(2170)
    
    def SR_1180_1080(self):
        """Simple Ratio of 1180 1080

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(1180)/self.R(1080)
    
    def SR_1180_1450(self):
        """Simple Ratio of 1180 1450

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(1180)/self.R(1450)
    
    def SR_1180_1675(self):
        """Simple Ratio of 1180 1675

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(1180)/self.R(1675)
    
    def SR_1180_2170(self):
        """Simple Ratio of 1180 2170

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(1180)/self.R(2170)
    
    def SR_1193_1126(self):
        """Simple Ratio of 1193 1126

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(1193)/self.R(1126)
    
    def SR_1250_1050(self):
        """Simple Ratio of 1250 1050

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(1250)/self.R(1050)
    
    def SR_1260_1080(self):
        """Simple Ratio of 1260 1080

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(1260)/self.R(1080)
    
    def SR_1260_1450(self):
        """Simple Ratio of 1260 1450

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(1260)/self.R(1450)
    
    def SR_1260_1675(self):
        """Simple Ratio of 1260 1675

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(1260)/self.R(1675)
    
    def SR_1260_2170(self):
        """Simple Ratio of 1260 2170

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(1260)/self.R(2170)
    
    def SR_1450_1080(self):
        """Simple Ratio of 1450 1080

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(1450)/self.R(1080)
    
    def SR_1450_1180(self):
        """Simple Ratio of 1450 1180

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(1450)/self.R(1180)
    
    def SR_1450_1260(self):
        """Simple Ratio of 1450 1260

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(1450)/self.R(1260)
    
    def SR_1450_960(self):
        """Simple Ratio of 1450 960

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(1450)/self.R(960)
    
    def SR_1600_820(self):
        """Simple Ratio of 1600 820

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(1600)/self.R(820)
    
    def SR_1650_2218(self):
        """Simple Ratio of 1650 2218

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(1650)/self.R(2218)
    
    def SR_1660_550(self):
        """Simple Ratio of 1660 550

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(1660)/self.R(550)
    
    def SR_1660_680(self):
        """Simple Ratio of 1660 680

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(1660)/self.R(680)
    
    def SR_1675_1080(self):
        """Simple Ratio of 1675 1080

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(1675)/self.R(1080)
    
    def SR_1675_1180(self):
        """Simple Ratio of 1675 1180

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(1675)/self.R(1180)
    
    def SR_1675_1260(self):
        """Simple Ratio of 1675 1260

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(1675)/self.R(1260)
    
    def SR_1675_960(self):
        """Simple Ratio of 1675 960

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(1675)/self.R(960)
    
    def SR_2170_1080(self):
        """Simple Ratio of 2170 1080

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(2170)/self.R(1080)
    
    def SR_2170_1180(self):
        """Simple Ratio of 2170 1180

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(2170)/self.R(1180)
    
    def SR_2170_1260(self):
        """Simple Ratio of 2170 1260

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(2170)/self.R(1260)
    
    def SR_2170_960(self):
        """Simple Ratio of 2170 960

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(2170)/self.R(960)
    
    def SR_430_680(self):
        """Simple Ratio of 430 680

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(430)/self.R(680)
    
    def SR_440_690(self):
        """Simple Ratio of 440 690

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(440)/self.R(690)
    
    def SR_440_740(self):
        """Simple Ratio of 440 740

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(440)/self.R(740)
    
    def BGI(self):
        """Blue green pigment index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(450)/self.R(550)
    
    def BRI(self):
        """Blue red pigment index

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(440)/self.R(690)
    
    def SR_520_420(self):
        """Simple Ratio of 520 420

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(520)/self.R(420)
    
    def SR_520_670(self):
        """Simple Ratio of 520 670

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(520)/self.R(670)
    
    def SR_520_760(self):
        """Simple Ratio of 520 760

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(520)/self.R(760)
    
    def SR_542_750(self):
        """Simple Ratio of 542 750

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(542)/self.R(750)
    
    def SR_550_420(self):
        """Simple Ratio of 550 420

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(550)/self.R(420)
    
    def SR_550_670(self):
        """Simple Ratio of 550 670

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(550)/self.R(670)
    
    def SR_550_680(self):
        """Simple Ratio of 550 680

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(550)/self.R(680)
    
    def SR_550_760(self):
        """Simple Ratio of 550 760

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(550)/self.R(760)
    
    def SR_550_800(self):
        """Simple Ratio of 550 800

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(550)/self.R(800)
    
    def SR_554_677(self):
        """Simple Ratio of 554 677

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(554)/self.R(677)
    
    def SR_556_750(self):
        """Simple Ratio of 556 750

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(556)/self.R(750)
    
    def SR_560_658(self):
        """Simple Ratio of 560 658

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(560)/self.R(658)
    
    def SR_605_420(self):
        """Simple Ratio of 605 420

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(605)/self.R(420)
    
    def SR_605_670(self):
        """Simple Ratio of 605 670

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(605)/self.R(670)
    
    def SR_605_760(self):
        """Simple Ratio of 605 760

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(605)/self.R(760)
    
    def SR_672_550(self):
        """Simple Ratio of 672 550

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(672)/self.R(550)
    
    def SR_672_708(self):
        """Simple Ratio of 672 708

        Returns
        -------
        numpy 2d-array
            A 2D numpy array of calcualted feature.
        """
        return self.R(672)/self.R(708)
    
    def SR_674_553(self):
        """Simple Ratio of 674 553

        Returns
        -------
        numpy 2d-array
            SR_674_553
        """
        return self.R(674)/self.R(553)
    
    def SR_675_555(self):
        """Simple Ratio of 675 555

        Returns
        -------
        numpy 2d-array
            SR_675_555
        """
        return self.R(675)/self.R(555)
    
    def SR_675_700(self):
        """Simple Ratio of 675 700

        Returns
        -------
        numpy 2d-array
            SR_675_700
        """
        return self.R(675)/self.R(700)
    
    def SR_678_750(self):
        """Simple Ratio of 678 750

        Returns
        -------
        numpy 2d-array
            SR_678_750
        """
        return self.R(678)/self.R(750)
    
    def SR_683_510(self):
        """Simple Ratio of 683 510

        Returns
        -------
        numpy 2d-array
            SR_683_510
        """
        return self.R(683)/self.R(510)
    
    def SR_685_735(self):
        """Simple Ratio of 685 735

        Returns
        -------
        numpy 2d-array
            SR_685_735
        """
        return self.R(685)/self.R(735)
    
    def SR_690_735(self):
        """Simple Ratio of 690 735

        Returns
        -------
        numpy 2d-array
            SR_690_735
        """
        return self.R(690)/self.R(735)
    
    def SR_690_740(self):
        """Simple Ratio of 690 740

        Returns
        -------
        numpy 2d-array
            SR_690_740
        """
        return self.R(690)/self.R(740)
    
    def SR_694_840(self):
        """Simple Ratio of 694 840

        Returns
        -------
        numpy 2d-array
            SR_694_840
        """
        return self.R(694)/self.R(840)
    
    def SR_695_420(self):
        """Simple Ratio of 695 420

        Returns
        -------
        numpy 2d-array
            SR_695_420
        """
        return self.R(695)/self.R(420)
    
    def SR_695_670(self):
        """Simple Ratio of 695 670

        Returns
        -------
        numpy 2d-array
            SR_695_670
        """
        return self.R(695)/self.R(670)
    
    def SR_695_760(self):
        """Simple Ratio of 695 760

        Returns
        -------
        numpy 2d-array
            SR_695_760
        """
        return self.R(695)/self.R(760)
    
    def SR_695_800(self):
        """Simple Ratio of 695 800

        Returns
        -------
        numpy 2d-array
            SR_695_800
        """
        return self.R(695)/self.R(800)
    
    def SR_700(self):
        """Simple Ratio of 700

        Returns
        -------
        numpy 2d-array
            SR_700
        """
        return 1/self.R(700)
    
    def SR_700_670(self):
        """Simple Ratio of 700 670

        Returns
        -------
        numpy 2d-array
            SR_700_670
        """
        return self.R(700)/self.R(670)
    
    def SR_705_722(self):
        """Simple Ratio of 705 722

        Returns
        -------
        numpy 2d-array
            SR_705_722
        """
        return self.R(705)/self.R(722)
    
    def SR_706_750(self):
        """Simple Ratio of 706 750

        Returns
        -------
        numpy 2d-array
            SR_706_750
        """
        return self.R(706)/self.R(750)
    
    def SR_710_420(self):
        """Simple Ratio of 710 420

        Returns
        -------
        numpy 2d-array
            SR_710_420
        """
        return self.R(710)/self.R(420)
    
    def SR_710_670(self):
        """Simple Ratio of 710 670

        Returns
        -------
        numpy 2d-array
            SR_710_670
        """
        return self.R(710)/self.R(670)
    
    def SR_710_760(self):
        """Simple Ratio of 710 760

        Returns
        -------
        numpy 2d-array
            SR_710_760
        """
        return self.R(710)/self.R(760)
    
    def SR_715_705(self):
        """Simple Ratio of 715 705

        Returns
        -------
        numpy 2d-array
            SR_715_705
        """
        return self.R(715)/self.R(705)
    
    def SR_730_706(self):
        """Simple Ratio of 730 706

        Returns
        -------
        numpy 2d-array
            SR_730_706
        """
        return self.R(730)/self.R(706)
    
    def SR_735_710(self):
        """Simple Ratio of 735 710

        Returns
        -------
        numpy 2d-array
            SR_735_710
        """
        return self.R(735)/self.R(710)
    
    def SR_740_720(self):
        """Simple Ratio of 740 720

        Returns
        -------
        numpy 2d-array
            SR_740_720
        """
        return self.R(740)/self.R(720)
    
    def SR_750_550(self):
        """Simple Ratio of 750 550

        Returns
        -------
        numpy 2d-array
            SR_750_550
        """
        return self.R(750)/self.R(550)
    
    def SR_750_555(self):
        """Simple Ratio of 750 555

        Returns
        -------
        numpy 2d-array
            SR_750_555
        """
        return self.R(750)/self.R(555)
    
    def SR_750_700(self):
        """Simple Ratio of 750 700

        Returns
        -------
        numpy 2d-array
            SR_750_700
        """
        return self.R(750)/self.R(700)
    
    def SR_750_705(self):
        """Simple Ratio of 750 705

        Returns
        -------
        numpy 2d-array
            SR_750_705
        """
        return self.R(750)/self.R(705)
    
    def SR_750_710(self):
        """Simple Ratio of 750 710

        Returns
        -------
        numpy 2d-array
            SR_750_710
        """
        return self.R(750)/self.R(710)
    
    def SR_752_690(self):
        """Simple Ratio of 752 690

        Returns
        -------
        numpy 2d-array
            SR_752_690
        """
        return self.R(752)/self.R(710)
    
    def Datt3(self):
        """Datt3, Simple Ratio of 7542 704

        Returns
        -------
        numpy 2d-array
            Datt3
        """
        return self.R(754)/self.R(704)
    
    def RARS(self):
        """Ratio Analysis of Reflectance Spectra, Simple Ratio of 760 500

        Returns
        -------
        numpy 2d-array
            RARS
        """
        return self.R(760)/self.R(500)
    
    def SR_760_695(self):
        """Simple Ratio of 760 695

        Returns
        -------
        numpy 2d-array
            SR_760_695
        """
        return self.R(760)/self.R(695)
    
    def SR_774_677(self):
        """Simple Ratio of 774 677

        Returns
        -------
        numpy 2d-array
            SR_774_677
        """
        return self.R(774)/self.R(677)
    
    def SR_800_1180(self):
        """Simple Ratio of 800 1180

        Returns
        -------
        numpy 2d-array
            SR_800_1180
        """
        return self.R(800)/self.R(1180)
    
    def SR_800_1280(self):
        """Simple Ratio of 800 1280

        Returns
        -------
        numpy 2d-array
            SR_800_1280
        """
        return self.R(800)/self.R(1280)
    
    def SR_800_1450(self):
        """Simple Ratio of 800 1450

        Returns
        -------
        numpy 2d-array
            SR_800_1450
        """
        return self.R(800)/self.R(1450)
    
    def SR_800_1660(self):
        """Simple Ratio of 800 1660

        Returns
        -------
        numpy 2d-array
            SR_800_1660
        """
        return self.R(800)/self.R(1660)
    
    def SR_800_1675(self):
        """Simple Ratio of 800 1675

        Returns
        -------
        numpy 2d-array
            SR_800_1675
        """
        return self.R(800)/self.R(1675)
    
    def SR_800_2170(self):
        """Simple Ratio of 800 2170

        Returns
        -------
        numpy 2d-array
            SR_800_2170
        """
        return self.R(800)/self.R(2170)
    
    def SR_800_470(self):
        """Simple Ratio of 800 470

        Returns
        -------
        numpy 2d-array
            SR_800_470
        """
        return self.R(800)/self.R(470)
    
    def SR_800_500(self):
        """Simple Ratio of 800 500

        Returns
        -------
        numpy 2d-array
            SR_800_500
        """
        return self.R(800)/self.R(500)
    
    def SR_800_550(self):
        """Simple Ratio of 800 550

        Returns
        -------
        numpy 2d-array
            SR_800_550
        """
        return self.R(800)/self.R(550)
    
    def SR_800_600(self):
        """Simple Ratio of 800 600

        Returns
        -------
        numpy 2d-array
            SR_800_600
        """
        return self.R(800)/self.R(600)
    
    def SR_800_635(self):
        """Simple Ratio of 800 635

        Returns
        -------
        numpy 2d-array
            SR_800_635
        """
        return self.R(800)/self.R(635)
    
    def SR_800_650(self):
        """Simple Ratio of 800 650

        Returns
        -------
        numpy 2d-array
            SR_800_650
        """
        return self.R(800)/self.R(650)
    
    def SR_800_670(self):
        """Simple Ratio of 800 670

        Returns
        -------
        numpy 2d-array
            SR_800_670
        """
        return self.R(800)/self.R(670)
    
    def SR_800_675(self):
        """Simple Ratio of 800 675

        Returns
        -------
        numpy 2d-array
            SR_800_675
        """
        return self.R(800)/self.R(675)
    
    def SR_800_680(self):
        """Simple Ratio of 800 680

        Returns
        -------
        numpy 2d-array
            SR_800_680
        """
        return self.R(800)/self.R(680)
    
    def SR_800_960(self):
        """Simple Ratio of 800 960

        Returns
        -------
        numpy 2d-array
            SR_800_960
        """
        return self.R(800)/self.R(960)
    
    def SR_800_550(self):
        """Simple Ratio of 800 550

        Returns
        -------
        numpy 2d-array
            SR_800_550
        """
        return self.R(800)/self.R(550)
    
    def SR_800_670(self):
        """Simple Ratio of 800 670

        Returns
        -------
        numpy 2d-array
            SR_800_670
        """
        return self.R(800)/self.R(670)
    
    def SR_810_560(self):
        """Simple Ratio of 810 560

        Returns
        -------
        numpy 2d-array
            SR_810_560
        """
        return self.R(810)/self.R(560)
    
    def SR_833_1649(self):
        """Simple Ratio of 833 1649

        Returns
        -------
        numpy 2d-array
            SR_833_1649
        """
        return self.R(833)/self.R(1649)
    
    def SR_833_658(self):
        """Simple Ratio of 833 658

        Returns
        -------
        numpy 2d-array
            SR_833_658
        """
        return self.R(833)/self.R(658)
    
    def SR_850_710(self):
        """Simple Ratio of 850 710

        Returns
        -------
        numpy 2d-array
            SR_850_710
        """
        return self.R(850)/self.R(710)
    
    def SR_850_1240(self):
        """Simple Ratio of 850 1240

        Returns
        -------
        numpy 2d-array
            SR_850_1240
        """
        return self.R(850)/self.R(1240)
    
    def SR_850_550(self):
        """Simple Ratio of 850 550

        Returns
        -------
        numpy 2d-array
            SR_850_550
        """
        return self.R(850)/self.R(550)
    
    def SR_850_708(self):
        """Simple Ratio of 850 708

        Returns
        -------
        numpy 2d-array
            SR_850_708
        """
        return self.R(850)/self.R(708)
    
    def SR_895_972(self):
        """Simple Ratio of 895 972

        Returns
        -------
        numpy 2d-array
            SR_895_972
        """
        return self.R(895)/self.R(972)
    
    def SR_900_680(self):
        """Simple Ratio of 900 680

        Returns
        -------
        numpy 2d-array
            SR_900_680
        """
        return self.R(900)/self.R(680)
    
    def SR_950_900(self):
        """Simple Ratio of 950 900

        Returns
        -------
        numpy 2d-array
            SR_950_900
        """
        return self.R(950)/self.R(900)
    
    def SR_960_1180(self):
        """Simple Ratio of 960 1180

        Returns
        -------
        numpy 2d-array
            SR_960_1180
        """
        return self.R(960)/self.R(1180)
    
    def SR_960_1260(self):
        """Simple Ratio of 960 1260

        Returns
        -------
        numpy 2d-array
            SR_960_1260
        """
        return self.R(960)/self.R(1260)
    
    def SR_960_1450(self):
        """Simple Ratio of 960 1450

        Returns
        -------
        numpy 2d-array
            SR_960_1450
        """
        return self.R(960)/self.R(1450)
    
    def SR_960_1675(self):
        """Simple Ratio of 960 1675

        Returns
        -------
        numpy 2d-array
            SR_960_1675
        """
        return self.R(960)/self.R(1675)
    
    def SR_960_2170(self):
        """Simple Ratio of 960 2170

        Returns
        -------
        numpy 2d-array
            SR_960_2170
        """
        return self.R(960)/self.R(2170)
    
    def PWI(self):
        """Plant Water Index, Simple Ratio of 970 900

        Returns
        -------
        numpy 2d-array
            PWI
        """
        return self.R(970)/self.R(900)
    
    def SR_970_902(self):
        """Simple Ratio of 960 2170

        Returns
        -------
        numpy 2d-array
            SR_970_902
        """
        return self.R(970)/self.R(902)
    
    def SRPI(self):
        """Simple Ratio Pigment Index

        Returns
        -------
        numpy 2d-array
            SRPI
        """
        return self.R(430)/self.R(680)
    
    def SR_355_365_gk(self):
        """Simple Ratio of SR_355_365_gk

        Returns
        -------
        numpy 2d-array
            SR_355_365_gk
        """
        return (self.R(355)/1.1)*self.R(365)
    
    def SAVI(self):
        """Soil Adjusted Vegetation Index

        Returns
        -------
        numpy 2d-array
            SR_355_365_gk
        """
        return 1.5*((self.R(800)-self.R(670))/(self.R(800)+self.R(670)+0.5))
    
    def SARVI2(self):
        """Soil and Atmospherically Resistant Vegetation Index 2

        Returns
        -------
        numpy 2d-array
            SR_355_365_gk
        """
        return 2.5*((self.R(860)-self.R(650))/(1+self.R(860)+6*self.R(650)-7.5*self.R(480)))
    
    def SAVI3(self):
        """Soil Adjusted Vegetation Index 3

        Returns
        -------
        numpy 2d-array
            SR_355_365_gk
        """
        return 1.5*((self.R(833)-self.R(658))/(self.R(833)+self.R(658)+0.5))
    
    def SBL(self):
        """Soil Background Line

        Returns
        -------
        numpy 2d-array
            SR_355_365_gk
        """
        return self.R(1000) - 2.4*self.R(650)
    
    def SLAVI(self):
        """Specific Leaf Area Vegetation Index

        Returns
        -------
        numpy 2d-array
            SLAVI
        """
        return self.R(860) / (self.R(650) + self.R(2200))
    
    def SPVI(self):
        """Spectral Polygon Vegetation Index

        Returns
        -------
        numpy 2d-array
            SPVI
        """
        return 0.4*(3.7*(self.R(800)-self.R(670))-1.2*np.abs(self.R(530)-self.R(670)))
    
    def SQRT_NIR_R(self):
        """Square root of NIR and Red

        Returns
        -------
        numpy 2d-array
            SQRT_NIR_R
        """
        return np.sqrt(self.R(860)/self.R(650))
    
    def SIPI1(self):
        """Structure Intensive Pigment Index 1

        Returns
        -------
        numpy 2d-array
            SIPI1
        """
        return (self.R(800)-self.R(445)) / (self.R(800)-self.R(680))
    
    def SIPI2(self):
        """Structure Intensive Pigment Index 2

        Returns
        -------
        numpy 2d-array
            SIPI2
        """
        return (self.R(800)-self.R(505)) / (self.R(800)-self.R(690))
    
    def SIPI3(self):
        """Structure Intensive Pigment Index 3

        Returns
        -------
        numpy 2d-array
            SIPI3
        """
        return (self.R(800)-self.R(470)) / (self.R(800)-self.R(680))
    
    def SBI(self):
        """Tasselled Cap - brightness

        Returns
        -------
        numpy 2d-array
            SBI
        """
        return 0.3037*self.R(520)+0.2793*self.R(600)+0.4743*self.R(690)+0.5585*self.R(900)+0.5082*self.R(1750)+0.1863*self.R(2350)
    
    def GVIMSS(self):
        """Tasselled Cap - Green Vegetation Index MSS

        Returns
        -------
        numpy 2d-array
            GVIMSS
        """
        return -0.283*self.R(600)-0.660*self.R(700)+0.577*self.R(800)+0.388*self.R(1100)
    
    def NSIMSS(self):
        """Tasselled Cap - Non Such Index MSS

        Returns
        -------
        numpy 2d-array
            NSIMSS
        """
        return -0.016*self.R(600)+0.131*self.R(700)-0.425*self.R(800)+0.882*self.R(1100)
    
    def SBIMSS(self):
        """Tasselled Cap - Soil Brightness Index MSS

        Returns
        -------
        numpy 2d-array
            NSIMSS
        """
        return 0.332*self.R(600)+0.603*self.R(700)+0.675*self.R(800)+0.262*self.R(1100)
    
    def GVI(self):
        """Tasselled Cap - vegetation

        Returns
        -------
        numpy 2d-array
            NSIMSS
        """
        return -0.284*self.R(520)-0.2435*self.R(600)-0.5436*self.R(690)+0.7243*self.R(900)+0.0840*self.R(1750)-0.1800*self.R(2350)
    
    def WET(self):
        """Tasselled Cap - wetness

        Returns
        -------
        numpy 2d-array
            NSIMSS
        """
        return 0.1509*self.R(520)+0.1973*self.R(600)+0.3279*self.R(690)+0.3406*self.R(900)-0.7112*self.R(1750)-0.4572*self.R(2350)
    
    def YVIMSS(self):
        """Tasselled Cap - Yellow Vegetation Index MSS

        Returns
        -------
        numpy 2d-array
            NSIMSS
        """
        return -0.899*self.R(600)+0.428*self.R(700)+0.076*self.R(800)-0.041*self.R(1100)
    
    def TCARI(self):
        """Transformed Chlorophyll Absorbtion Ratio

        Returns
        -------
        numpy 2d-array
            NSIMSS
        """
        return 3*((self.R(700)-self.R(670))-0.2*(self.R(700)-self.R(550))*(self.R(700)/self.R(670)))
    
    def TCARI1510(self):
        """Transformed Chlorophyll Absorbtion Ratio 1510

        Returns
        -------
        numpy 2d-array
            NSIMSS
        """
        return 3*((self.R(700)-self.R(1510))-0.2*(self.R(700)-self.R(550))*(self.R(700)/self.R(1510)))
    
    def TCARI2(self):
        """Transformed Chlorophyll Absorbtion Ratio 2

        Returns
        -------
        numpy 2d-array
            NSIMSS
        """
        return 3*((self.R(750)-self.R(705))-0.2*(self.R(750)-self.R(550))*(self.R(750)/self.R(705)))
    
    def TNDVI(self):
        """Transformed NDVI

        Returns
        -------
        numpy 2d-array
            NSIMSS
        """
        return np.sqrt(((self.R(860)-self.R(650))/(self.R(860)+self.R(650)))+0.5)
    
    def TSAVI(self):
        """Transformed Soil Adjusted Vegetation Index

        Returns
        -------
        numpy 2d-array
            NSIMSS
        """
        a, s, X = 0.33, 0.5, 1.5
        return (s*(self.R(860)-s*self.R(650)-a))/(a*self.R(860)+self.R(650)-a*s+X*(1+s**2))
    
    def TVI(self):
        """Transformed Vegetation Index

        Returns
        -------
        numpy 2d-array
            NSIMSS
        """
        return np.sqrt(self.NDVI+0.5)
    
    def TCI(self):
        """Triangular chlorophyll index

        Returns
        -------
        numpy 2d-array
            NSIMSS
        """
        return 1.2*(self.R(700)-self.R(550)) - 1.5*(self.R(670)-self.R(550))*np.sqrt(self.R(700)/self.R(670))
    
    def TGI(self):
        """Triangular greenness index

        Returns
        -------
        numpy 2d-array
            NSIMSS
        """
        return -0.5*(190*(self.R(670)-self.R(550))-120*(self.R(670)-self.R(480)))
    
    def TVI(self):
        """Triangular vegetation index

        Returns
        -------
        numpy 2d-array
            NSIMSS
        """
        return 0.5*(120*(self.R(750)-self.R(550))-200*(self.R(670)-self.R(550)))
    
    def VARIgreen(self):
        """Visible Atmospherically Resistant Index Green

        Returns
        -------
        numpy 2d-array
            NSIMSS
        """
        return (self.R(565)-self.R(680))/(self.R(565)+self.R(680)-self.R(490))
    
    def VARI700(self):
        """Visible Atmospherically Resistant Index 700

        Returns
        -------
        numpy 2d-array
            NSIMSS
        """
        return (self.R(700)-1.7*self.R(680)+0.7*self.R(490))/(self.R(700)+2.3*self.R(680)-1.3*self.R(490))
    
    def VARIrededge(self):
        """Visible Atmospherically Resistant Index RedEdge

        Returns
        -------
        numpy 2d-array
            NSIMSS
        """
        return (self.R(710)-self.R(680))/(self.R(710)+self.R(680))
    
    def WDRVI(self):
        """Wide Dynamic Range Vegetation Index

        Returns
        -------
        numpy 2d-array
            NSIMSS
        """
        return (0.1*self.R(860) - self.R(650)) / (0.1*self.R(860) + self.R(650))
    
#Source = https://www.indexdatabase.de/db/i.php?offset=1    
    
    
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
                #print(e)
                pass
        if len(not_calculated_features) > 0:
            print(f'{len(not_calculated_features)} features could not be calculated.')
        
        return None