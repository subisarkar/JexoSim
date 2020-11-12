''' 
JexoSim
2.0
Generate transmission files
v1.0

'''

import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
import csv
import os
import jexosim

def run(database_path):
   
    jexosim_path =  os.path.dirname((os.path.dirname(jexosim.__file__)))

    dfp = '%s/jexosim/data/JWST'%(jexosim_path)
    if not os.path.exists(dfp):
            os.makedirs(dfp)       
   
    #==============================================================================
    # dictionaries
    #==============================================================================
    
    MIRI = {
            'ch':'MIRI',
            'file': None
            }
    
    NIRSpec =    {
            'ch':'NIRSpec',
            'file': 'jwst/nirspec/detector/jwst_nirspec_ipckernel_20160902193401.fits'}
        
    NIRISS = {
            'ch':'NIRISS',
            'file': 'jwst/detector/hawaii_2rg_ipc_kernel.fits'}    
        
    NIRCam = {
            'ch':'NIRCam',
            'file': 'jwst/nircam/detector/jwst_nircam_h2rg_ipckernel_20160902164019.fits'}
              
    
        
    #==============================================================================
    # Compile the files into JexoSim compatible .csv files and put in right folder
    #==============================================================================
   
    # choose which channels to compile
    dic_list = [NIRSpec, NIRISS, NIRCam]
    
    for dic in dic_list:
         
        filename = '%s/%s'%(database_path, dic['file'])
        hdul = fits.open(filename)
        #try:
        #    hdul = fits.open(filename)
        #except IOError:
        #    hdul = fits.open(filename.lower())
 
        
        a= '%s/%s/IPC_kernel'%(dfp, dic['ch'])
        if not os.path.exists(a):
            os.makedirs(a)
        
        if dic['ch'] == 'NIRCam':
            kernel = hdul[1].data
        else:
            kernel = hdul[0].data

        np.save('%s/kernel.npy'%(a), kernel)
         
    
    
 
    
