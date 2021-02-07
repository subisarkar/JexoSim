"""
JexoSim
2.0
Systematics module
v1.01
    
"""

from jexosim.lib import jexosim_lib, systematics_lib
from jexosim.lib.jexosim_lib import jexosim_msg
import copy

def run(opt):
    
    opt = systematics_lib.gen_prnu_grid(opt)
    opt = systematics_lib.gen_systematic_grid(opt)
    
    opt.qe_original = copy.deepcopy(opt.qe)
    opt.qe_uncert_original = copy.deepcopy(opt.qe_uncert)
    opt.syst_grid_original = copy.deepcopy(opt.syst_grid)
 
    return opt

      
    
