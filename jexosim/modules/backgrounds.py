"""
JexoSim
2.0
Backgrounds modules
v1.0

"""


from jexosim.lib.jexosim_lib import jexosim_msg, jexosim_plot
from jexosim.lib import jexosim_lib, backgrounds_lib 
import numpy  as np
from astropy import units as u
import scipy.interpolate
import scipy.signal
import copy

def run(opt):

      jexosim_msg('Running backgrounds module ...\n ', opt.diagnostics)
      
      opt.zodi, zodi_transmission  = backgrounds_lib.zodical_light(opt)
      opt.emission = backgrounds_lib.optical_emission(opt)
      zodi_transmission.rebin(opt.star.sed.wl)
      jexosim_lib.sed_propagation(opt.star.sed, zodi_transmission)
      ch = opt.channel
      Omega_pix = 2.0*np.pi*(1.0-np.cos(np.arctan(0.5/ch.camera.wfno_x())))*u.sr
      Apix = ((ch.detector_pixel.pixel_size.val).to(u.m))**2           
      opt.zodi.sed *= Apix*Omega_pix*opt.Re *u.electron/u.W/u.s 
      opt.emission.sed *= Apix*Omega_pix*opt.Re *u.electron/u.W/u.s 
      jexosim_lib.sed_propagation(opt.zodi, opt.total_transmission)
      opt.zodi.rebin(opt.x_wav_osr)
      opt.emission.rebin(opt.x_wav_osr)   
      opt.zodi.sed     *= opt.d_x_wav_osr 
      opt.emission.sed *= opt.d_x_wav_osr
      zodi_photons_sed = copy.deepcopy(opt.zodi.sed)
      emission_photons_sed = copy.deepcopy(opt.emission.sed)
      if ch.camera.slit_width.val == 0:
          slit_size =  opt.fpn[1]*2  # to ensure all wavelengths convolved onto all pixels in slitless case
      else:
          slit_size =  ch.camera.slit_width.val    
      # need to work on this to make more accurate : filter wavelength range might not correspond to wl sol on the detector, i.e. some wavelengths not being included that should be
      opt.zodi.sed     = scipy.signal.convolve(opt.zodi.sed, 
		      np.ones(np.int(slit_size*ch.simulation_factors.osf())), 
		      'same') * opt.zodi.sed.unit
      opt.emission.sed = scipy.signal.convolve(opt.emission.sed.value, 
		      np.ones(np.int(slit_size*ch.simulation_factors.osf())), 
		      'same') * opt.emission.sed.unit
      opt.zodi_sed_original = copy.deepcopy(opt.zodi.sed)
      opt.emission_sed_original = copy.deepcopy(opt.emission.sed)     
      jexosim_plot('zodi spectrum', opt.diagnostics, xdata = opt.zodi.wl, ydata=opt.zodi.sed)
      jexosim_plot('emission spectrum', opt.diagnostics, xdata = opt.emission.wl, ydata=opt.emission.sed)   
      # ====Now work out quantum yield ==============================================
      zodi_w_qy =  zodi_photons_sed*opt.quantum_yield.sed
      zodi_w_qy = scipy.signal.convolve(zodi_w_qy, 
		      np.ones(np.int(slit_size*ch.simulation_factors.osf())), 
		      'same') * opt.zodi.sed.unit    
      opt.qy_zodi = zodi_w_qy  / opt.zodi.sed      
      emission_w_qy =  emission_photons_sed*opt.quantum_yield.sed
      emission_w_qy = scipy.signal.convolve(emission_w_qy, 
		      np.ones(np.int(slit_size*ch.simulation_factors.osf())), 
		      'same') * opt.emission.sed.unit    
      opt.qy_emission = emission_w_qy  / opt.emission.sed   
 
      opt.qy_zodi[np.isnan(opt.qy_zodi)] = 1
      opt.qy_emission[np.isnan(opt.qy_emission)] = 1
      

 
      return opt
      

    
  