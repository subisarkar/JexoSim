"""
A re-usable PRNU grid is generated, and large enough to be used for all channels.
Alternately a random grid can be created each time by activating the ApplyRandomPRNU option
in the common configuration file
"""
import numpy as np
import os
import jexosim

 
def run(rms= 0.03, rms_uncert=0.005):
    
    jexosim_path =  os.path.dirname((os.path.dirname(jexosim.__file__)))

    dfp = '%s/jexosim/data2/JWST/PRNU'%(jexosim_path)
    
    if not os.path.exists(dfp):
        os.makedirs(dfp) 

    x = 5000
    y = 5000
        
    qe = np.ones((y,x)) + np.random.normal(0,rms,(y,x))
    qe_uncert = np.ones((y,x)) + np.random.normal(0,rms_uncert,(y,x))
    
    np.save('%s/qe_rms.npy'%dfp, qe)
    np.save('%s/qe_uncert.npy'%dfp, qe_uncert)   
    


