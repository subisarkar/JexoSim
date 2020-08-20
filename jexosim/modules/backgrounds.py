"""
JexoSim
2.0
Backgrounds modules
v1.0

"""

from jexosim.classes.sed import Sed
from jexosim.lib.jexosim_lib import jexosim_msg
from jexosim.lib import jexosim_lib 
import numpy           as np
from astropy import units as u
import scipy.interpolate
import scipy.signal
import copy

def zodical_light(opt):
   
    deg = opt.fake_exosystem.ecliptic_lat.val.value
    wl = opt.common.common_wl
    jexosim_msg('Zodi model initiated...\n', opt.diagnostics)
    jexosim_msg('Ecliptic latitude in deg: %s   \n'%(deg), opt.diagnostics)
    deg = abs(deg)
 
    if deg >=57.355:
        level = 1.0
    else:
        level = -0.22968868*(np.log10(deg+1))**7 + 1.12162927*(np.log10(deg+1))**6 - 1.72338015*(np.log10(deg+1))**5 + 1.13119022*(np.log10(deg+1))**4 - 0.95684987*(np.log10(deg+1))**3+ 0.2199208*(np.log10(deg+1))**2- 0.05989941*(np.log10(deg+1))  +2.57035947
    jexosim_msg('Zodi model coefficient... %s\n'%(level), opt.diagnostics)
    spectrum = level*(3.5e-14*jexosim_lib.planck(wl, 5500*u.K) +
                      jexosim_lib.planck(wl, 270*u.K) * 3.58e-8)
    sed = Sed(wl, spectrum )
    transmission = Sed(wl, np.ones(wl.size)*u.dimensionless_unscaled)
    zodi = [sed, transmission]

    return zodi
    
def emission_estimation(N, temp, emissivity, transmission):
   
    t_i = transmission.sed**(1./N) # estimate transmission of each optical surface 
    for i in range(0,N): #loop through each surface i=0 to i = N-1
        if i == 0 :
             emission_sed =  emissivity*jexosim_lib.planck(transmission.wl, temp)*t_i**(N-1-i)
        else:
            emission_sed = emission_sed + emissivity*jexosim_lib.planck(transmission.wl, temp)*t_i**(N-1-i)
        idx = np.argwhere(transmission.sed==0) 
        emission_sed[idx] = 0.0* emission_sed.unit            
            
    emission=  Sed(transmission.wl, emission_sed) 
      
    return emission
    
    
def optical_emission(opt):
    #telescope     
      N = np.int(opt.common_optics.emissions.optical_surface.no_surfaces)
      temp = opt.common_optics.emissions.optical_surface.val
      emissivity = np.float(opt.common_optics.emissions.optical_surface.emissivity) 
      
      opt.telescope_emission = emission_estimation(N, temp, emissivity, opt.telescope_transmission)
      jexosim_lib.sed_propagation(opt.telescope_emission,opt.channel_transmission)
                   
    #channel
      N = np.int(opt.channel.emissions.optical_surface.no_surfaces)
      temp = opt.channel.emissions.optical_surface.val
      emissivity = np.float(opt.channel.emissions.optical_surface.emissivity)  
      opt.channel_emission = emission_estimation(N, temp, emissivity, opt.channel_transmission)
      
      emission = Sed(opt.channel_transmission.wl, opt.telescope_emission.sed+ opt.channel_emission.sed )

      return emission    
    

def run(opt):

      jexosim_msg('Running backgrounds module ...\n ', opt.diagnostics)
      
      opt.zodi, zodi_transmission  = zodical_light(opt)
      opt.emission = optical_emission(opt)
      
      zodi_transmission.rebin(opt.star.sed.wl)
      jexosim_lib.sed_propagation(opt.star.sed,zodi_transmission)

      ch = opt.channel
            
      Omega_pix = 2.0*np.pi*(1.0-np.cos(np.arctan(0.5/ch.camera.wfno_x())))*u.sr
      Apix = ((ch.detector_pixel.pixel_size.val).to(u.m))**2           

      opt.zodi.sed*= Apix*Omega_pix*opt.Re *u.electron/u.W/u.s 
      opt.emission.sed *= Apix*Omega_pix*opt.Re *u.electron/u.W/u.s   
      
      jexosim_lib.sed_propagation(opt.zodi, opt.total_transmission)
      
      opt.zodi.rebin(opt.x_wav_osr)
      opt.emission.rebin(opt.x_wav_osr)   

      opt.zodi.sed     *= opt.d_x_wav_osr 
      opt.emission.sed *= opt.d_x_wav_osr
       
      # need to work on this to make more accurate : filter wavelength range might not correspond to wl sol on the detector, i.e. some wavelengths not being included that should be
      opt.zodi.sed     = scipy.signal.convolve(opt.zodi.sed, 
		      np.ones(np.int(ch.camera.slit_width()*ch.simulation_factors.osf())), 
		      'same') * opt.zodi.sed.unit
      opt.emission.sed = scipy.signal.convolve(opt.emission.sed, 
		      np.ones(np.int(ch.camera.slit_width()*ch.simulation_factors.osf())), 
		      'same') * opt.emission.sed.unit
        
      opt.zodi_sed_original = copy.deepcopy(opt.zodi.sed)
      opt.emission_sed_original = copy.deepcopy(opt.emission.sed)
                 
      return opt
      

    
  