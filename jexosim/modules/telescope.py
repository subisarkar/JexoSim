"""
JexoSim 
2.0
Telescope module
v1.0

"""

from jexosim.classes.sed import Sed
from jexosim.lib import jexosim_lib
from jexosim.lib.jexosim_lib import jexosim_msg
import numpy           as np
from astropy import units as u
import copy 
 
def run(opt):

      jexosim_msg('tel check 1: %s'%(opt.star.sed.sed.max()), opt.diagnostics)
      opt.star_sed = copy.deepcopy(opt.star.sed) #copy of star flux at telescope
         
      opt.Aeff = 0.25*np.pi*opt.common_optics.telescope_effective_diameter()**2
      opt.star.sed.sed*= opt.Aeff
      jexosim_msg('tel check 2: %s'%(opt.star.sed.sed.max()), opt.diagnostics)
      tr_ =np.array([1.]*len(opt.common.common_wl))*u.dimensionless_unscaled
    
      opt.common_optics.transmissions.optical_surface = opt.common_optics.transmissions.optical_surface \
          if isinstance(opt.common_optics.transmissions.optical_surface, list) \
          else [opt.common_optics.transmissions.optical_surface]    
      for op in opt.common_optics.transmissions.optical_surface:
          dtmp=np.loadtxt(op.transmission.replace('__path__', opt.__path__), delimiter=',')
          tr = Sed(dtmp[:,0]*u.um,dtmp[:,1]*u.dimensionless_unscaled)
          tr.rebin(opt.common.common_wl)  
          tr_ *=tr.sed
          
      opt.telescope_transmission = Sed(opt.common.common_wl, tr_)          
      jexosim_lib.sed_propagation(opt.star.sed, opt.telescope_transmission)
      jexosim_msg('tel check 3: %s'%(opt.star.sed.sed.max()), opt.diagnostics)
      return opt
  