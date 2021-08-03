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
      opt.emission  = backgrounds_lib.optical_emission(opt) 
      opt.sunshield = backgrounds_lib.sunshield_emission(opt)
    
      # import matplotlib.pyplot as plt
      # plt.figure ('backgrounds 1')
      # plt.plot(opt.sunshield.wl, opt.sunshield.sed, 'r', label='sunshield before transmission')
      # plt.plot(opt.zodi.wl, opt.zodi.sed, 'g--', label='zodi before transmission')
      # plt.plot(opt.emission.wl, opt.emission.sed ,'bx', label = 'optical after transmission')
      # plt.grid()
      # plt.legend()

      # propagate star through zodi transmission
      jexosim_lib.sed_propagation(opt.star.sed, zodi_transmission)
       
      ch = opt.channel
      Omega_pix = 2.0*np.pi*(1.0-np.cos(np.arctan(0.5/ch.camera.wfno_x())))*u.sr
      Apix = ((ch.detector_pixel.pixel_size.val).to(u.m))**2     
      
      opt.zodi.sed *= Apix*Omega_pix*opt.Re *u.electron/u.W/u.s 
      opt.sunshield.sed *= Apix*Omega_pix*opt.Re *u.electron/u.W/u.s 
      opt.emission.sed *= Apix*Omega_pix*opt.Re *u.electron/u.W/u.s 
      
      # apply transmission to zodi and sunshield (not to optical emission -alreay applied)
      jexosim_lib.sed_propagation(opt.zodi, opt.total_transmission)
      jexosim_lib.sed_propagation(opt.sunshield, opt.total_transmission)
      
     
      opt.zodi.sed[np.isnan(opt.zodi.sed)] = 0
      opt.sunshield.sed[np.isnan(opt.sunshield.sed)] = 0
      opt.emission.sed[np.isnan(opt.emission.sed)] = 0
      
            
      opt.zodi.sed     *= opt.d_x_wav_osr 
      opt.sunshield.sed     *= opt.d_x_wav_osr 
      opt.emission.sed *= opt.d_x_wav_osr
   
      zodi_photons_sed = copy.deepcopy(opt.zodi.sed)
      sunshield_photons_sed = copy.deepcopy(opt.sunshield.sed)
      emission_photons_sed = copy.deepcopy(opt.emission.sed)
      
      if ch.camera.slit_width.val == 0:
          slit_size =  opt.fpn[1]*2  # to ensure all wavelengths convolved onto all pixels in slitless case
      else:
          slit_size =  ch.camera.slit_width.val   
          
      # import matplotlib.pyplot as plt
      # plt.figure ('backgrounds 2') 
      # plt.plot(opt.zodi.wl, opt.zodi.sed, 'bo-')
      # print (opt.zodi.sed)
       
      # need to work on this to make more accurate : filter wavelength range might not correspond to wl sol on the detector, i.e. some wavelengths not being included that should be
      opt.zodi.sed     = scipy.signal.convolve(opt.zodi.sed, 
		      np.ones(np.int(slit_size* ch.simulation_factors.osf())), 
		      'same') * opt.zodi.sed.unit
      opt.sunshield.sed     = scipy.signal.convolve(opt.sunshield.sed, 
		      np.ones(np.int(slit_size*ch.simulation_factors.osf())), 
		      'same') * opt.sunshield.sed.unit 
      opt.emission.sed = scipy.signal.convolve(opt.emission.sed.value, 
		      np.ones(np.int(slit_size*ch.simulation_factors.osf())), 
		      'same') * opt.emission.sed.unit
      
      # print (opt.zodi.sed)
      # import matplotlib.pyplot as plt
      # plt.figure ('backgrounds 2') 
      # plt.plot(opt.zodi.wl, opt.zodi.sed, 'ro-')
      # xxxx
      
      
      opt.zodi_sed_original = copy.deepcopy(opt.zodi.sed)
      opt.sunshield_sed_original = copy.deepcopy(opt.sunshield.sed) 
      opt.emission_sed_original = copy.deepcopy(opt.emission.sed)  
      
      jexosim_plot('zodi spectrum', opt.diagnostics, xdata = opt.zodi.wl, ydata=opt.zodi.sed)
      jexosim_plot('sunshield spectrum', opt.diagnostics, xdata = opt.sunshield.wl, ydata=opt.sunshield.sed)
      jexosim_plot('emission spectrum', opt.diagnostics, xdata = opt.emission.wl, ydata=opt.emission.sed)   
      
      
      # ====Now work out quantum yield ==============================================
      # weight the qy by relative number of photons
      zodi_w_qy =  zodi_photons_sed*opt.quantum_yield.sed
      # convolve this with slit
      zodi_w_qy = scipy.signal.convolve(zodi_w_qy, 
		      np.ones(np.int(slit_size*ch.simulation_factors.osf())), 
		      'same') * opt.zodi.sed.unit    
      # divide by convolved spectrum.... thus this factor will return the expected qy
      opt.qy_zodi = zodi_w_qy  / opt.zodi.sed 
      
      # (Ph conv. slit )x Qy'  = (Ph x Qy) conv. slit
      # thus Qy' is the needed Qy per pixel applied to the convolve Ph spectrum.
      
      sunshield_w_qy =  sunshield_photons_sed*opt.quantum_yield.sed
      sunshield_w_qy = scipy.signal.convolve(sunshield_w_qy, 
		      np.ones(np.int(slit_size*ch.simulation_factors.osf())), 
		      'same') * opt.sunshield.sed.unit    
      opt.qy_sunshield = sunshield_w_qy  / opt.sunshield.sed 
      
      emission_w_qy =  emission_photons_sed*opt.quantum_yield.sed
      emission_w_qy = scipy.signal.convolve(emission_w_qy, 
		      np.ones(np.int(slit_size*ch.simulation_factors.osf())), 
		      'same') * opt.emission.sed.unit    
      opt.qy_emission = emission_w_qy  / opt.emission.sed   
 
      opt.qy_zodi[np.isnan(opt.qy_zodi)] = 1
      opt.qy_sunshield[np.isnan(opt.qy_sunshield)] = 1
      opt.qy_emission[np.isnan(opt.qy_emission)] = 1
      

      return opt
      

    
  