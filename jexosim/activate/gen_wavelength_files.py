''' 
JexoSim 
2.0
Generate wavelength solutions
v1.0

'''

import numpy as np
from astropy.io import fits
import csv
import os
import jexosim


def run(database_path):
        
    jexosim_path =  os.path.dirname((os.path.dirname(jexosim.__file__)))

    dfp = '%s/jexosim/data/JWST'%(jexosim_path)
    if not os.path.exists(dfp):
            os.makedirs(dfp) 
   
    
    NIRSpec = {
            'ch':'NIRSpec',
            'ch0': 'nirspec',
            'grism':['G395M','G395H','G235H', 'G235M','G140H', 'G140M','PRISM'],
            'file': ['jwst_nirspec_g395m_disp_20160902193401.fits',
            'jwst_nirspec_g395h_disp_20160902193401.fits', 'jwst_nirspec_g235h_disp_20160902193401.fits',
            'jwst_nirspec_g235m_disp_20160902193401.fits','jwst_nirspec_g140h_disp_20160902193401.fits',
            'jwst_nirspec_g140m_disp_20160902193401.fits','jwst_nirspec_prism_disp_20160902193401.fits'],
            'wav_limits': [[2.87,5.1], [2.87,5.14], [1.66,3.05], [1.66,3.07], [0.97,1.82], [0.97,1.84], [0.6,5.6]],
            'pix_size':18.0
            }
    
    
    MIRI = {
            'ch':'MIRI',
            'ch0': 'miri',
            'grism':['prism'],
            'file': ['jwst_miri_p750l_disp_20170404135013.fits'],
            'wav_limits': [[4.0,18.0]],
            'pix_size':25.0
            }
            
    NIRCam = {
            'ch':'NIRCam',
            'ch0': 'nircam',
            'grism':['F444W_GRISM','F322W2_GRISM'],
            'file': ['jwst_nircam_disp_20170901102005.fits', 'jwst_nircam_disp_20170901102005.fits'],
            'wav_limits': [[3.7,5.0], [2.42, 4.15]],
            'pix_size':18.0        
            }        
            
    
    NIRISS = {
            'ch':'NIRISS',
            'ch0': 'niriss',
            'grism':['SOSS_GRISM'],
            'file': ['jwst_niriss_soss-256-ord1_trace_20160929180722.fits'],
            'wav_limits': [[0.9, 2.8]],
            'pix_size':18.0
            }    
    
    # wavelength ranges from Lustig-Yeager et al. (2019)   
    #==============================================================================
    # Compile the files into JexoSim compatible .csv files and put in right folder
    #==============================================================================
    
    # choose which channels to compile
    dic_list = [MIRI, NIRSpec, NIRISS, NIRCam]
    
    # dic_list = [NIRCam]


    for dic in dic_list:
        for i in range(len(dic['file']))   :

            if dic['ch']!= 'NIRISS':
                
                wav_max = dic['wav_limits'][i][1]
                wav_min = dic['wav_limits'][i][0]
                 
                filepath = '%s/jwst/%s/dispersion/%s'%(database_path, dic['ch0'], dic['file'][i])
                # hdul = fits.open(filepath)
              
                try:
                    hdul = fits.open(filepath)
                except IOError:
                    hdul = fits.open(filepath.lower())          
            
                wl = hdul[1].data['WAVELENGTH']
                dlds = hdul[1].data['DLDS']
                
                wl0=wl[0]
                wlpix = []

                for j in range (10000): # gives the wavelengt on each pixel (increasing in steps of 1 pixel)
                    wlpix.append(wl0)
                    dlds0 = np.interp(wl0, wl, dlds) #rate of change of wavelength with pixel
                    wl0 = wl0+ dlds0           
                                                          
                x = np.arange(0,len(wlpix)*dic['pix_size'],dic['pix_size']) 
                x= x[::-1]
                
                wl_mid = (wav_max+ wav_min)/2.0
                x_mid  = np.interp(wl_mid, wlpix, x)
                x = x - x_mid
                
                # import matplotlib.pyplot as plt                
                # plt.figure('wav sol gen check')
                # plt.plot(x, wlpix)
            
                y = np.zeros(len(x))
                
            
            elif dic['ch'] == 'NIRISS':
                filepath = '%s/jwst/%s/wavepix/%s'%(database_path, dic['ch0'], dic['file'][i])
                hdul = fits.open(filepath)
                
                wl = hdul[1].data['WAVELENGTH']

                x_pix = hdul[1].data['DETECTOR_PIXELS']
                y_pix = hdul[1].data['TRACE']
                
                x = x_pix*dic['pix_size']
                x = x - (x.max()-x.min())/2. 
                x= x[::-1]
                
                y = y_pix*dic['pix_size']
                y = y - (y.max()-y.min())/2. 
                y= y[::-1]
                
                wlpix = wl

            header = ["# lambda [um]","y [micron]","x [micron]"]
            label = '%s_%s_dispersion'%(dic['ch'],dic['grism'][i])

            a= '%s/%s/Wavelengths'%(dfp, dic['ch'])
            if not os.path.exists(a):
                os.makedirs(a)      
            b= '%s/%s.csv'%(a, label)

            with open(b, 'w') as csvfile:
                filewriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                filewriter.writerow(header)                        
                for i in range(len(wlpix)):
                    filewriter.writerow([wlpix[i], y[i], x[i]])
 
 
 
 
