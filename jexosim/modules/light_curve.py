"""
JexoSim 
2.0
Light curve module
v1.1

"""

import numpy as np
from astropy import units as u
from jexosim.lib.jexosim_lib import jexosim_msg, jexosim_plot
from jexosim.lib import exosystem_lib
import copy

def run(opt):
    
  opt.cr    =  opt.planet.sed.sed[opt.offs::opt.osf]
  opt.cr_wl =  opt.planet.sed.wl[opt.offs::opt.osf]      
  if opt.timeline.apply_lc.val ==1:                     
     # Apply lightcurve model
    opt.cr    =  opt.planet.sed.sed[opt.offs::opt.osf]
    opt.cr_wl =  opt.planet.sed.wl[opt.offs::opt.osf]  
    # inputs needed to use pytransit : z no longer calculated as separate step  
    t0 = (opt.time_at_transit).to(u.day).value.item()
    per =  (opt.planet.planet.P).to(u.day).value
    ars = ( (opt.planet.planet.a).to(u.m)/ (opt.planet.planet.star.R).to(u.m)).value
    inc = (opt.planet.planet.i).to(u.rad).value
    ecc = opt.planet.planet.e    
    omega = 0.0     
     
    if opt.timeline.apply_lo_dens_LC.val == 1:
        timegrid = (opt.ndr_end_time).to(u.day).value
        opt.z_params = [timegrid, t0, per, ars, inc, ecc, omega]
        opt.lc, opt.ldc = exosystem_lib.get_light_curve(opt, opt.cr, opt.cr_wl, 					 		 
        							opt.observation.obs_type.val)
        
    else:  # use high density timegrid, and average for each NDR step
        highD_timegrid  = (np.arange(opt.frame_time, opt.total_observing_time+ opt.frame_time/2, opt.frame_time)*u.s).to(u.day).value 
        opt.z_params = [highD_timegrid, t0, per, ars, inc, ecc, omega]
        highD_timegrid  = np.arange(opt.frame_time, opt.total_observing_time+ opt.frame_time/2, opt.frame_time)*u.s
        highD_lc, opt.ldc = exosystem_lib.get_light_curve(opt, opt.cr, opt.cr_wl, 					 		 
        							opt.observation.obs_type.val)
    
        lc = np.hstack((highD_lc, highD_lc[:,0].reshape(highD_lc.shape[0],1)*0))
        idx = np.hstack(([0],np.cumsum(opt.ndr_sequence).astype(int)))   
        lc0 = np.add.reduceat(lc,idx, axis = 1)[:,:-1] /opt.ndr_sequence 
        lc0[lc0==np.inf] = 0
        opt.lc = lc0
           
  else:
      opt.lc = 0
      opt.z = 0
      opt.ldc = 0
      
  opt.cr_wl_original = copy.deepcopy(opt.cr_wl)
  opt.cr_original = copy.deepcopy(opt.cr)
  opt.lc_original = copy.deepcopy(opt.lc)
  opt.ldc_original = copy.deepcopy(opt.ldc)
     
  return opt        
 


          

