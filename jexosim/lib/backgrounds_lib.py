"""
JexoSim 2.0

Backgrounds library

"""

import numpy as np
from jexosim.lib import jexosim_lib
from jexosim.lib.jexosim_lib import jexosim_msg, planck
from astropy import units as u
from jexosim.classes.sed import Sed


def zodical_light(opt):
   
    deg = opt.model_exosystem.ecliptic_lat.val.value
    wl = opt.x_wav_osr
    jexosim_msg('Zodi model initiated...\n', opt.diagnostics)
    jexosim_msg('Ecliptic latitude in deg: %s   \n'%(deg), opt.diagnostics)
    deg = abs(deg)
    if deg >=57.355:
        level = 1.0
    else:
        level = -0.22968868*(np.log10(deg+1))**7 + 1.12162927*(np.log10(deg+1))**6 - 1.72338015*(np.log10(deg+1))**5 + 1.13119022*(np.log10(deg+1))**4 - 0.95684987*(np.log10(deg+1))**3+ 0.2199208*(np.log10(deg+1))**2- 0.05989941*(np.log10(deg+1))  +2.57035947
    jexosim_msg('Zodi model coefficient... %s\n'%(level), opt.diagnostics)
    spectrum = level*(3.5e-14*planck(wl, 5500*u.K) +
                      planck(wl, 270*u.K) * 3.58e-8)
    sed = Sed(wl, spectrum )
    transmission = Sed(wl, np.ones(wl.size)*u.dimensionless_unscaled)
    zodi = [sed, transmission]
    return zodi

def sunshield_emission(opt):
 
    wl = opt.x_wav_osr
    # polynomial fit to publically available data
    z = [-9.95238428e-15,  1.25072156e-12, -6.35949515e-11,  1.67141033e-09,
         -2.42822718e-08,  1.98344621e-07, -8.56711400e-07,  1.52392307e-06]    
    r=7; y =0
    for i in range (0,r+1):
        y = y + z[i]*wl.value**(r-i)   
    idx = np.argwhere(wl.value < 7)  
    y[idx] = 0 
    y *= u.W/u.m**2/u.um/u.sr         
    sed = Sed(wl, y)
    return sed

    
def emission_estimation(N, temp, emissivity, transmission):
 
    t_i = transmission.sed**(1./N) # estimate transmission of each optical surface 
    for i in range(0,N): #loop through each surface i=0 to i = N-1
        if i == 0 :
             emission_sed =  emissivity*planck(transmission.wl, temp)*t_i**(N-1-i)
        else:
            emission_sed = emission_sed + emissivity*planck(transmission.wl, temp)*t_i**(N-1-i)
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
     # combine 
      emission = Sed(opt.channel_transmission.wl, opt.telescope_emission.sed+ opt.channel_emission.sed )
      return emission  
 
  
# def primary_mirror_emission(opt):
#     #telescope
#       wl = opt.x_wav_osr 
#       # wl = np.linspace(5,30,1000)*u.um
#       temp = opt.common_optics.emissions.optical_surface.val
#       emissivity = np.float(opt.common_optics.emissions.optical_surface.emissivity)
#       emission  =  emissivity*planck(wl, temp) 
#       sed = Sed(wl, emission)
#       return sed
      
  
# def optical_emission_new(opt):
#     #telescope     
#       N = np.int(opt.common_optics.emissions.optical_surface.no_surfaces)
#       temp = opt.common_optics.emissions.optical_surface.val
#       emissivity = np.float(opt.common_optics.emissions.optical_surface.emissivity) 
#       wl = opt.x_wav_osr
#       total_trans  = opt.telescope_transmission.sed
#       tr_per_surface =  total_trans**(1/N)
#       E = np.zeros((N, len(total_trans)))
#       for i in range(1,N+1):
#           E[i-1] = planck(wl, temp)*emissivity * tr_per_surface**(N-i)
#       telescope_emission = E.sum(axis=0) 
#       telescope_emission[np.isnan(telescope_emission)] = 0
    
#       telescope_emission = Sed(wl, telescope_emission)
#       jexosim_lib.sed_propagation(telescope_emission, opt.channel_transmission)
#      # channel   
#       N = np.int(opt.channel.emissions.optical_surface.no_surfaces)
#       temp = opt.channel.emissions.optical_surface.val
#       emissivity = np.float(opt.channel.emissions.optical_surface.emissivity)
#       wl = opt.x_wav_osr
#       total_trans  = opt.channel_transmission.sed
#       tr_per_surface =  total_trans**(1/N)
#       E = np.zeros((N, len(total_trans)))
#       for i in range(1,N+1):
#           E[i-1] = planck(wl, temp)*emissivity * tr_per_surface**(N-i)
#       channel_emission = E.sum(axis=0) 
#       channel_emission[np.isnan(channel_emission)] = 0

#       channel_emission = Sed(wl, channel_emission)
      
#       total_emission = Sed(wl, channel_emission.sed + telescope_emission.sed)
      
   
#       return total_emission
 
 
    