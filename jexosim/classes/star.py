
import numpy   as np
from astropy.io import fits
from astropy import units as u
from jexosim.classes import sed
from astropy import constants as const
import os, sys
from jexosim.lib.jexosim_lib import jexosim_msg, planck
import jexosim

class Star():

  def __init__(self, opt):
      
    self.opt = opt
    
    # opt.exosystem.star.d = 4.48966476375e+17*u.m
           
    jexosim_msg ("Star temperature:  %s"%(opt.exosystem.star.T), opt.diagnostics)
    jexosim_msg ("Star name %s, dist %s, radius %s"%(opt.exosystem.star.name, opt.exosystem.star.d, opt.exosystem.star.R), opt.diagnostics)  
   
    jexosim_path =  os.path.dirname((os.path.dirname(jexosim.__file__)))
    databases_dir = '%s/archive'%(jexosim_path)   
    cond=0
    for root, dirs, files in os.walk(databases_dir):
        for dirc in dirs:
            if 'BT-Settl' in dirc:
                dirc_name = dirc
                cond=1
                break
    if cond==0:
        print ('Error: database not found')    
    sed_folder = '%s/%s'%(databases_dir, dirc_name)

   
    if opt.exosystem_params.star_spectrum_model.val == 'simple':
          ph_wl, ph_sed = self.get_simple_spectrum (np.linspace(0.4,12.0,int(1e6))*u.um, 		
    			    opt.exosystem.star.T)			   
          jexosim_msg ("Using black body spectrum for star...", opt.diagnostics)
    else:      
          ph_wl, ph_sed = self.read_phoenix_spectrum(sed_folder, 					 
						    opt.exosystem.star.T, 
						    opt.exosystem.star.logg, 
						    opt.exosystem.star.Z)  
              
    if opt.exosystem_params.star_spectrum_mag_norm.val == 1:
        ph_sed  = self.useTelFlux(ph_wl, ph_sed, opt.exosystem_params.star_spectrum_mag_band.val, 
                                  opt.exosystem_params.star_spectrum_mag.val)
    else  :        
        ph_sed  *=  ((opt.exosystem.star.R).to(u.m)/(opt.exosystem.star.d).to(u.m))**2 		      # [W/m^2/mu]
    
    jexosim_msg("star check 2: %s"%ph_sed.max(), self.opt.diagnostics)
    self.sed = sed.Sed(ph_wl, ph_sed)
    
          
  def read_phoenix_spectrum(self, sed_folder, star_temperature, star_logg, star_f_h):

    jexosim_msg ("Star... star_temperature %s, star_logg %s, star_f_h %s"%(star_temperature, star_logg, star_f_h) , self.opt.diagnostics)

    if star_temperature.value >= 2400:
        t0 = np.round(star_temperature,-2)/100.
    else:
        t0= np.round(np.round(star_temperature,-1)/100./0.5)*0.5
    
    logg0 =  np.round(np.round(star_logg,1)/0.5)*0.5
    
    cond=0     
    for root, dirs, files in os.walk(sed_folder):
        for filename in files:
            if filename == 'lte0%s-%s-0.0a+0.0.BT-Settl.spec.fits'%(t0.value, logg0):
                ph_file = '%s/%s'%(sed_folder, filename)
                cond = 1
            elif filename == 'lte0%s-%s-0.0a+0.0.BT-Settl.spec.fits.gz'%(t0.value, logg0):
                ph_file = '%s/%s'%(sed_folder, filename)
                jexosim_msg ("Star file used:  %s"%(ph_file) , self.opt.diagnostics)
                cond = 1
    if cond==0:
        jexosim_msg ("Error:  no star file found", 1)
        sys.exit()
                 
    with fits.open(ph_file) as hdu:
        wl = u.Quantity(hdu[1].data.field('Wavelength'),
                     hdu[1].header['TUNIT1']).astype(np.float64)
        sed = u.Quantity(hdu[1].data.field('Flux'), u.W/u.m**2/u.um).astype(np.float64)
        if hdu[1].header['TUNIT2'] != 'W / (m2 um)': print ('Exception')
                         
     #remove duplicates
        idx = np.nonzero(np.diff(wl))
        wl = wl[idx]
        sed = sed[idx]
        hdu.close()
 
    jexosim_msg("star check 1: %s"%sed.max(), self.opt.diagnostics)
    return wl, sed
    
  def get_simple_spectrum(self, wl,  star_temperature):
    sed =  np.pi*planck(wl, star_temperature) * u.sr
    return wl, sed

  def useTelFlux(self, wl, sed,  mag_type='J', mag=8.0):
      
      jexosim_msg ("normalizing stellar spectrum to %s mag %s"%(mag_type, mag),  self.opt.diagnostics)
      
      wl = wl.value
      if mag_type == 'J':
          wav = 1.235
          Jy = 1594
          
          #wav = 1.25
          #Jy = 1603  
          
      if mag_type == 'K':

          wav = 2.22
          Jy = 667                      
          
      # F_0 = scipy.constants.c*1e6*1e-26*(Jy)/wav**2
      F_0 = const.c.value*1e6*1e-26*(Jy)/wav**2
 
      F_x = F_0*10**(-mag/2.5)
                
      xx=10000

      box = np.ones(xx)/xx
      sed0 = np.convolve(sed,box,'same')  

      idx = np.argwhere(wl>=wav)[0].item()
      F_s = sed0[idx]
      
      scale = F_x/F_s
      sed= sed*scale
  
      return sed
    

 
