"""
JexoSim 
2.0
Systematics module
v1.0

"""
import numpy as np
from jexosim.lib import jexosim_lib
from jexosim.lib.jexosim_lib import jexosim_msg
import copy

def run(opt):
    
    if opt.noise.ApplyRandomPRNU.val == 1:
          opt.qe = np.random.normal(1, 0.01*opt.noise.sim_prnu_rms.val, opt.fp_original[1::3,1::3].shape) # for random uncertainty
          opt.qe_uncert = np.random.normal(1, 0.01*opt.noise.sim_flat_field_uncert.val, opt.fp_original[1::3,1::3].shape) # for random uncertainty  
          jexosim_msg ("RANDOM PRNU GRID SELECTED...",  opt.diagnostics)
    else:
          opt.qe = np.load('%s/data/JWST/PRNU/qe_rms.npy'%(opt.__path__))[0:opt.fp_original [1::3,1::3].shape[0],0:opt.fp_original[1::3,1::3].shape[1]]
          opt.qe_uncert = np.load('%s/data/JWST/PRNU/qe_uncert.npy'%(opt.__path__))[0:opt.fp_original[1::3,1::3].shape[0],0:opt.fp_original[1::3,1::3].shape[1]]      
          jexosim_msg ("PRNU GRID SELECTED FROM FILE...", opt.diagnostics)

    opt.qe_original = copy.deepcopy(opt.qe)
    opt.qe_uncert_original = copy.deepcopy(opt.qe_uncert)

    return opt

 
 
      
    
