"""
JexoSim 
2.0
Telescope module
v1.0

"""

from jexosim.classes.sed import Sed
from jexosim.lib import jexosim_lib, instrument_lib
from jexosim.lib.jexosim_lib import jexosim_msg
import numpy as np
from astropy import units as u
import copy 

 
def run(opt):
    
      opt.osf         = np.int(opt.channel.simulation_factors.osf())
      opt.offs        = np.int(opt.channel.simulation_factors.pix_offs())    
      fpn = opt.channel.detector_array.array_geometry.val.split(',')
      opt.fpn = [int(fpn[0]), int(fpn[1])]  
      opt.fp  = np.zeros(( int(opt.fpn[0]*opt.channel.simulation_factors.osf.val),
                            int(opt.fpn[1]*opt.channel.simulation_factors.osf.val) ))    
      opt.fp_signal  = np.zeros(( int(opt.fpn[0]*opt.channel.simulation_factors.osf.val),
                            int(opt.fpn[1]*opt.channel.simulation_factors.osf.val) ))
      opt.fp_it  = np.zeros(( int(opt.fpn[0]*opt.channel.simulation_factors.osf.val),
                            int(opt.fpn[1]*opt.channel.simulation_factors.osf.val) ))    
      opt.fp_signal_it  = np.zeros(( int(opt.fpn[0]*opt.channel.simulation_factors.osf.val),
                            int(opt.fpn[1]*opt.channel.simulation_factors.osf.val) ))
     
      opt.fp_delta = opt.channel.detector_pixel.pixel_size.val / opt.channel.simulation_factors.osf.val                     
#      opt.x_wav_osr, opt.x_pix_osr, opt.y_pos_osr = instrument_lib.usePoly(opt)
      opt.x_wav_osr, opt.x_pix_osr, opt.y_pos_osr = instrument_lib.useInterp(opt)
      opt.R = instrument_lib.getR(opt)   
      jexosim_msg('tel check 1: %s'%(opt.star.sed.sed.max()), opt.diagnostics)
      opt.star_sed = copy.deepcopy(opt.star.sed) #copy of star flux at telescope
      opt.star_sed2 = copy.deepcopy(opt.star.sed) #copy of star flux at telescope
      opt.Aeff = 0.25*np.pi*opt.common_optics.telescope_effective_diameter()**2 
      opt.star.sed.sed*= opt.Aeff
      opt.star.sed_it.sed*= opt.Aeff
      
  
      jexosim_msg('tel check 2: %s'%(opt.star.sed.sed.max()), opt.diagnostics)
      tr_ =np.array([1.]*len(opt.x_wav_osr))*u.dimensionless_unscaled
      opt.common_optics.transmissions.optical_surface = opt.common_optics.transmissions.optical_surface \
          if isinstance(opt.common_optics.transmissions.optical_surface, list) \
          else [opt.common_optics.transmissions.optical_surface]    
      for op in opt.common_optics.transmissions.optical_surface:
          dtmp=np.loadtxt(op.transmission.replace('__path__', opt.__path__), delimiter=',')
          tr = Sed(dtmp[:,0]*u.um,dtmp[:,1]*u.dimensionless_unscaled)
          tr.rebin(opt.x_wav_osr)  
          tr_ *=tr.sed       
      opt.telescope_transmission = Sed(opt.x_wav_osr, tr_)                   
      opt.star.sed.rebin(opt.x_wav_osr) 
      opt.star.sed_it.rebin(opt.x_wav_osr)  
      jexosim_msg('tel check 2a: %s'%(opt.star.sed.sed.max()), opt.diagnostics)
  
      opt.planet.sed.rebin(opt.x_wav_osr) 
      jexosim_lib.sed_propagation(opt.star.sed, opt.telescope_transmission)
      jexosim_lib.sed_propagation(opt.star.sed_it, opt.telescope_transmission)
      jexosim_msg('tel check 3: %s'%(opt.star.sed.sed.max()), opt.diagnostics)

      dtmp=np.loadtxt(opt.channel.detector_array.quantum_yield().replace('__path__', opt.__path__), delimiter=',')
      quantum_yield = Sed(dtmp[:,0]*u.um, dtmp[:,1]*u.dimensionless_unscaled)
      quantum_yield.rebin(opt.x_wav_osr)
      quantum_yield.sed = np.where(quantum_yield.sed==0,1,quantum_yield.sed) # avoids quantum yield of 0 due to interpolation issue 
      opt.quantum_yield = quantum_yield 
     
      
      return opt
  
