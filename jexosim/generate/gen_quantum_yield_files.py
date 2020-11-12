''' 
JexoSim
2.0
Generate quantum yield files
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
            'file': ['jwst/miri/qe/jwst_miri_imager_qe_20170404135013.fits'],
            'label': ['MIRI_quantum_yield'],
            'wl_min':4.5, 'wl_max':20.1, 'wl_del':0.001
            }
    
    NIRSpec =    {
        'ch':'NIRSpec',
        'file': ['jwst/nirspec/qe/jwst_nirspec_qe_20160902193401.fits'],    
        'label': ['NIRSpec_quantum_yield'],
        'wl_min':0.4, 'wl_max':6.0, 'wl_del':0.001
        }
        
    NIRISS = {
            'ch':'NIRISS',
            'file': ['jwst/niriss/qe/jwst_niriss_h2rg_qe_20160902163017.fits'],
            'label': ['NIRISS_quantum_yield'],
            'wl_min':0.6, 'wl_max':3.0, 'wl_del':0.001
            }    
        
    NIRCam = {
            'ch':'NIRCam',
            'file': ['jwst/nircam/qe/jwst_nircam_lw_qe_20160902164019.fits'],
            'label': ['NIRCam_LW_quantum_yield'],
            'wl_min':2.2, 'wl_max':5.5, 'wl_del':0.001
            }     
    
        
    #==============================================================================
    # Compile the files into JexoSim compatible .csv files and put in right folder
    #==============================================================================
   
    # choose which channels to compile
    dic_list = [MIRI, NIRSpec, NIRISS, NIRCam]
    
    for dic in dic_list:
        
        for i in range (len(dic['file']))   :
           
            WL = np.arange(dic['wl_min'],dic['wl_max'], dic['wl_del'])
            TR = 1.0

            filename = '%s/%s'%(database_path, dic['file'][i])
            hdul = fits.open(filename)
            #try:
            #    hdul = fits.open(filename)
            #except IOError:
            #    hdul = fits.open(filename.lower())
            
            wl = hdul[1].data['WAVELENGTH']
            tr = hdul[1].data['CONVERSION']
            tr0 = interp1d(wl,tr, kind = 'linear', bounds_error = False)(WL)
            TR = TR*tr0

            idx = np.argwhere(np.isnan(tr0))
            WL = np.delete(WL,idx)
            tr0 = np.delete(tr0,idx)
            
            a= '%s/%s/Transmissions'%(dfp, dic['ch'])
            if not os.path.exists(a):
                os.makedirs(a) 
        
            b= '%s/%s.csv'%(a, dic['label'][i])
            with open(b, 'w') as csvfile:
                filewriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for i in range(len(tr0)):
                
                   filewriter.writerow([WL[i], tr0[i]])

    
    
 
    
