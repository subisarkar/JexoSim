"""
JexoSim
2.0
JDP Jitter decorrelation module
v1.0

"""

from scipy.interpolate import interp2d, interp1d
from astropy import units as u
import numpy as np
from jexosim.lib.jexosim_lib import jexosim_msg

class JexoSimDecorr:

    def __init__(self, data, opt):
        
        self.pixArr = np.arange(0, data.shape[0])
        self.nExp =  data.shape[2]
        self.mAccum = opt.effective_multiaccum        
        self.pointingNdr = opt.pointing_timeline[:,0]
        self.pointingJittX = opt.pointing_timeline[:,1]
        self.pointingJittY = opt.pointing_timeline[:,2]      
        self.pointing_timeline = opt.pointing_timeline
        self.plate_scale =  (opt.channel.detector_pixel.plate_scale_x.val).to(u.deg).value
        self.modelBeam = data[...,0]
        self.calcModelPointingOffsets()
      
    def calcModelPointingOffsets(self):
        spec=[]
        spat = []        
        for j in range(self.mAccum-1 , self.mAccum*self.nExp, self.mAccum):
            spec_offset = self.pointingJittX[self.pointingNdr==j]
            spat_offset = self.pointingJittY[self.pointingNdr==j]
            spec_offset = np.mean(spec_offset) / self.plate_scale
            spat_offset = np.mean(spat_offset) / self.plate_scale         
            spec.append(spec_offset)
            spat.append(spat_offset)
                
        self.modelJitterOffsetSpec = np.array(spec)        
        self.modelJitterOffsetSpat = np.array(spat)
        self.modelJitterOffsetSpec = self.modelJitterOffsetSpec - self.modelJitterOffsetSpec.mean()
        self.modelJitterOffsetSpat = self.modelJitterOffsetSpat - self.modelJitterOffsetSpat.mean()          
       
    def getPointingOffsets(self):      
      XX = {'spec':self.modelJitterOffsetSpec, 'spat':self.modelJitterOffsetSpat  }         
      return XX
  
    
class JitterRemoval:
    @staticmethod
    def cubicIterp(jdc, jiggOffsetMeasure, data):
        upsampleFactor =1 
        shiftedMaps =[]
        for i in range(jdc.nExp):
                  map2 = data[...,i]                  
                  f = interp2d(np.arange(map2.shape[1]), np.arange(map2.shape[0]), map2, kind='cubic')
                  tempMap2Cubic = f(np.linspace(0, map2.shape[1]-1, map2.shape[1]*upsampleFactor) -jiggOffsetMeasure['spec'][i], np.linspace(0, map2.shape[0]-1, map2.shape[0]*upsampleFactor) -jiggOffsetMeasure['spat'][i])
                  tempMap2CubicLR = tempMap2Cubic#                  
        
                  shiftedMaps.append(tempMap2CubicLR)
        return shiftedMaps; del shiftedMaps
    @staticmethod 
    def fftShift(jdc,jiggOffsetMeasure, data):
        shiftedMaps =[]
        for i in range(jdc.nExp):
              im = data[...,i]		
              dy = jiggOffsetMeasure['spec'][i]
              dx = jiggOffsetMeasure['spat'][i]
              imFft = np.fft.fftshift(np.fft.fft2(im))
              xF,yF = np.meshgrid(np.arange(im.shape[0]) - im.shape[0]/2,np.arange(im.shape[1]) - im.shape[1]/2)
              imFft=imFft*np.exp(-1j*2*np.pi*(xF.T*dx/im.shape[0]+yF.T*dy/im.shape[1]))
              shiftedIm = np.fft.ifft2(np.fft.ifftshift(imFft))
              shiftedMaps.append(np.abs(shiftedIm))  
        return shiftedMaps           


def crossCorr1d(refData, testData, xCorWin, type_):
    xCor = np.zeros(2*xCorWin)
    for j in range(2*xCorWin):
        xCor[j] = np.std((refData[xCorWin:-xCorWin] - testData[j:-2*xCorWin+j])**2)
    xInterp = np.linspace(0,len(xCor)-1, 30000)
    f2 = interp1d(np.arange(len(xCor)), xCor , kind='cubic') 
    offsetPx = xInterp[np.argmin(f2(xInterp))]
    return offsetPx

def getRelativeOffsets(imageRef, imgeTest):
    # Collapse Spatial Axis
    tempRef = np.sum(imageRef,1)
    tempTest = np.sum(imgeTest, 1)
    tempRef/=np.sum(tempRef)
    tempTest/=np.sum(tempTest)
    xCorWin = 26
    xCorWin = 10;
    offsetSpatial = -crossCorr1d(tempRef, tempTest, xCorWin, 'spat')+xCorWin
    # Collapse Spectral Axis
    tempRef = np.sum(imageRef,0)
    tempTest = np.sum(imgeTest, 0)
    tempRef/=np.sum(tempRef)
    tempTest/=np.sum(tempTest)
    xCorWin = 26
    xCorWin = 10;  
    offsetSpec = -crossCorr1d(tempRef, tempTest, xCorWin, 'spec')+xCorWin
    return {'spec':offsetSpec, 'spat':offsetSpatial}

class jitterCode():
    
    def __init__(self, data, opt):
        
        method = opt.channel.data_pipeline.jitterMethod.val    
        self.opt = opt
        self.data = data
        self.jdc = JexoSimDecorr(self.data, self.opt)
        
                
        if method =='xcorr-interp' or method == 'xcorr-fft':         
            jiggOffsetMeasure = {'spec':np.zeros(self.jdc.nExp), 'spat':np.zeros(self.jdc.nExp)}
            for i in range(self.jdc.nExp):
                im= data[...,i]
                offset = getRelativeOffsets(self.jdc.modelBeam, im)           
                jiggOffsetMeasure['spec'][i] = offset['spec']
                jiggOffsetMeasure['spat'][i] = offset['spat']                                                   

            if method =='xcorr-interp':
                self.shiftedMaps = JitterRemoval.cubicIterp(self.jdc, jiggOffsetMeasure, data)
            elif  method == 'xcorr-fft':
                self.shiftedMaps = JitterRemoval.fftShift(self.jdc, jiggOffsetMeasure, data) # does not need pointing info or pscale
        
        elif method =='pointing-interp' or method == 'pointing-fft':
            if method =='pointing-interp':
                jiggOffsetMeasure  = self.jdc.getPointingOffsets()
  
                self.shiftedMaps = JitterRemoval.cubicIterp(self.jdc, self.jdc.getPointingOffsets(), self.data)
            if method =='pointing-fft':
                self.shiftedMaps = JitterRemoval.fftShift(self.jdc, self.jdc.getPointingOffsets(), self.data)

        jexosim_msg ("method %s"%(method), opt.diagnostics)
    
        self.getData()
    
    def getData(self):    
        for i in range(self.data.shape[2]):
            self.data[...,i] =  self.shiftedMaps[i]
        self.decorrData = self.data






