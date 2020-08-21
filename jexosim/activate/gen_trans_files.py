''' Generates transmission files for use in JexoSim from the Pandeia database.
Please credit Pandeia (see Github) if using this database'''

import numpy as np
from astropy.io import fits
from scipy.interpolate import interp1d
import csv
import os
import jexosim

def run(params):
    
    database_path = params['database_pandeia']
    
    jexosim_path =  os.path.dirname((os.path.dirname(jexosim.__file__)))

    dfp = '%s/jexosim/data/JWST'%(jexosim_path)
    if not os.path.exists(dfp):
            os.makedirs(dfp) 
            
   
    #==============================================================================
    # dictionaries
    #==============================================================================
    
    OTE = {
            'ch':'OTE',
            'file': ['jwst/telescope/jwst_telescope_ote_thruput.fits'],
            'label': ['OTE_trans'],
            'wl_min':0.4, 'wl_max':30.0, 'wl_del':0.001
            }
             
    MIRI = {
            'ch':'MIRI',
            'file': ['jwst/miri/blaze/jwst_miri_p750l_speceff_20161121185659.fits',
            'jwst/miri/qe/jwst_miri_imager_qe_20170404135013.fits',
            'jwst/miri/qe/jwst_miri_imager_qe_20170404135013.fits',
            'jwst/miri/qe/jwst_miri_imager_qe_20170404135013.fits'],
            'label': ['MIRI_prism_trans','MIRI_QE', 'MIRI_internal_trans', 'MIRI_contamination'],
            'wl_min':4.5, 'wl_max':20.1, 'wl_del':0.001
            }
    
    NIRSpec =    {
        'ch':'NIRSpec',
        'file': [
        'jwst/nirspec/qe/jwst_nirspec_qe_20160902193401.fits',
        'jwst/nirspec/optical/jwst_nirspec_mos_internaloptics_throughput_20160902193401.fits',
        'jwst/nirspec/filters/jwst_nirspec_f290lp_trans_20160902193401.fits',
        'jwst/nirspec/blaze/jwst_nirspec_g395M_speceff_20160902193401.fits',    
            
        'jwst/nirspec/filters/jwst_nirspec_f070lp_trans_20160902193401.fits',    
        'jwst/nirspec/filters/jwst_nirspec_f100lp_trans_20160902193401.fits',   
        'jwst/nirspec/filters/jwst_nirspec_f170lp_trans_20160902193401.fits',    
        'jwst/nirspec/filters/jwst_nirspec_clear_trans_20160902193401.fits',       
        'jwst/nirspec/blaze/jwst_nirspec_g140M_speceff_20160902193401.fits',    
        'jwst/nirspec/blaze/jwst_nirspec_g140H_speceff_20160902193401.fits',    
        'jwst/nirspec/blaze/jwst_nirspec_g235M_speceff_20160902193401.fits',    
        'jwst/nirspec/blaze/jwst_nirspec_g235H_speceff_20160902193401.fits',
        'jwst/nirspec/blaze/jwst_nirspec_prism_speceff_20160902193401.fits'
        ],    
        'label': [
        'NIRSpec_MOS_internal_optics_trans',
        'NIRSpec_QE',
        'NIRSpec_F290LP_trans',
        'NIRSpec_G395M_trans',   
            
        'NIRSpec_F070LP_trans',
        'NIRSpec_F100LP_trans',
        'NIRSpec_F170LP_trans',
        'NIRSpec_clear_trans',   
        'NIRSpec_G140M_trans',
        'NIRSpec_G140H_trans',
        'NIRSpec_G235M_trans',
        'NIRSpec_G235H_trans',
        'NIRSpec_prism_trans'    
        ],
        'wl_min':0.4, 'wl_max':6.0, 'wl_del':0.001
        }
        
    NIRISS = {
            'ch':'NIRISS',
            'file': [
            'jwst/niriss/blaze/jwst_niriss_gr700xd-ord1_speceff_20160927172406.fits',
            'jwst/niriss/optical/jwst_niriss_internaloptics_throughput_20170417172726.fits',
            'jwst/niriss/qe/jwst_niriss_h2rg_qe_20160902163017.fits',
            'jwst/niriss/qe/jwst_niriss_h2rg_qe_20160902163017.fits'
            ],
            'label': [
            'NIRISS_grism_trans',
            'NIRISS_internal_trans',
            'NIRISS_QE',
            'NIRISS_mechanical_loss'
            ],
            'wl_min':0.6, 'wl_max':3.0, 'wl_del':0.001
            }    
        
    NIRCam = {
            'ch':'NIRCam',
            'file': [
            'jwst/nircam/filters/jwst_nircam_f444w_trans_20160902164019.fits',
            'jwst/nircam/filters/jwst_nircam_f322w2_trans_20160902164019.fits',
            'jwst/nircam/qe/jwst_nircam_lw_qe_20160902164019.fits',
            'jwst/nircam/optical/jwst_nircam_internaloptics_throughput_20170216143448.fits',
            'jwst/nircam/optical/jwst_nircam_lw_dbs_20160923153727.fits',
            'jwst/nircam/blaze/jwst_nircam_speceff_20160902164019.fits'
            ],
            'label': [
            'NIRCam_F444W_trans',
            'NIRCam_F322W2_trans',
            'NIRCam_LW_QE',
            'NIRCam_internal_trans',
            'NIRCam_LW_DBS_trans',
            'NIRCam_grism_trans'
            ],
            'wl_min':2.2, 'wl_max':5.5, 'wl_del':0.001
            }     
    
        
    #==============================================================================
    # Compile the files into JexoSim compatible .csv files and put in right folder
    #==============================================================================
   
    # choose which channels to compile
    dic_list = [OTE, MIRI, NIRSpec, NIRISS, NIRCam]
    
    for dic in dic_list:
        
        for i in range (len(dic['file']))   :
           
            WL = np.arange(dic['wl_min'],dic['wl_max'], dic['wl_del'])
            TR = 1.0

            filename = '%s/%s'%(database_path, dic['file'][i])
            hdul = fits.open(filename)
            
            wl = hdul[1].data['WAVELENGTH']
            tr = hdul[1].data['THROUGHPUT']
            tr0 = interp1d(wl,tr, kind = 'linear', bounds_error = False)(WL)
            TR = TR*tr0
            
            if dic == MIRI:
                if i ==2:
                    tr0=tr0*0 + 0.982**7 #for miri internal optics transmission 
                if i ==3:
                    tr0=tr0*0 + 0.8 #for miri contamination 
            if dic == NIRISS:         
                if i ==3:
                    tr0=tr0*0 + 0.67 #Montreal website says 33% mechanical loss
        
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

    
    
 
    
