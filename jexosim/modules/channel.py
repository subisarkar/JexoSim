"""
JexoSim
2.0
Channel module
v1.0

"""

from jexosim.classes.sed import Sed
from jexosim.lib import jexosim_lib 
from jexosim.lib.jexosim_lib import jexosim_msg, jexosim_plot
import numpy           as np
from astropy import units as u
from astropy import constants as const
import copy
 
def run(opt):
    
      jexosim_msg('channel check0 : %s'%(opt.star.sed.sed.max()), opt.diagnostics)
      tr_ =np.array([1.]*len(opt.x_wav_osr))*u.dimensionless_unscaled
      opt.channel.transmissions.optical_surface = opt.channel.transmissions.optical_surface if isinstance(opt.channel.transmissions.optical_surface, list) else \
                                                                      [opt.channel.transmissions.optical_surface]                                                              
      for op in opt.channel.transmissions.optical_surface:
          dtmp=np.loadtxt(op.transmission.replace('__path__', opt.__path__), delimiter=',')
          tr = Sed(dtmp[:,0]*u.um, dtmp[:,1]*u.dimensionless_unscaled)
          tr.rebin(opt.x_wav_osr)
          tr_ *=tr.sed            
      opt.channel_transmission = Sed(opt.x_wav_osr, tr_) 
      opt.total_transmission =   Sed(opt.channel_transmission.wl, opt.telescope_transmission.sed*opt.channel_transmission.sed )    
      jexosim_lib.sed_propagation(opt.star.sed, opt.channel_transmission) 
      jexosim_lib.sed_propagation(opt.star.sed_it, opt.channel_transmission)  
      # apply QE and convert to electrons
      dtmp=np.loadtxt(opt.channel.detector_array.qe().replace('__path__', opt.__path__), delimiter=',')
      qe = Sed(dtmp[:,0]*u.um, dtmp[:,1]*u.dimensionless_unscaled)
      qe.rebin(opt.x_wav_osr)
      PCE = Sed(opt.total_transmission.wl, opt.total_transmission.sed*qe.sed)
      TotTrans = Sed(opt.total_transmission.wl, opt.total_transmission.sed)
      opt.PCE = PCE.sed
      opt.TotTrans = TotTrans.sed
      jexosim_plot('PCE', opt.diagnostics, xdata = PCE.wl, ydata = PCE.sed)    
      jexosim_plot('Total transmission not including QE', opt.diagnostics, xdata = TotTrans.wl, ydata = TotTrans.sed)
      opt.qe_spec = qe
      opt.Re = qe.sed * (qe.wl).to(u.m)/(const.c.value * const.h.value * u.m)
      opt.star.sed.sed *= opt.Re*u.electron/u.W/u.s
      opt.star.sed_it.sed *= opt.Re*u.electron/u.W/u.s
      jexosim_plot('channel star sed check', opt.diagnostics, 
                   xdata = opt.x_wav_osr, ydata=opt.star.sed.sed, marker='-' )        
      jexosim_msg('check 1.3 - Star sed max:  %s'%(opt.star.sed.sed.max()), opt.diagnostics)    
      jexosim_plot('Wavelength solution check -oversampled pixels', opt.diagnostics, 
                   xdata = opt.x_pix_osr, ydata =opt.x_wav_osr, marker='o-' )
      jexosim_plot('Wavelength solution check -normal pixels', opt.diagnostics, 
                   xdata = opt.x_pix_osr[1::3], ydata =opt.x_wav_osr[1::3], marker='o-' )                 
      opt.d_x_wav_osr = np.zeros_like(opt.x_wav_osr)
      idx = np.where(opt.x_wav_osr > 0.0)
      opt.d_x_wav_osr[idx] = np.gradient(opt.x_wav_osr[idx])
      if np.any(opt.d_x_wav_osr < 0): opt.d_x_wav_osr *= -1.0
      jexosim_plot('D x wav osr check', opt.diagnostics, 
                   xdata = opt.x_pix_osr, ydata = opt.d_x_wav_osr, marker='o')
      jexosim_plot('D x wav osr check 2', opt.diagnostics, 
                   xdata = opt.x_wav_osr, ydata = opt.d_x_wav_osr, marker='o')        
      jexosim_msg ("check 1.4:  %s"%(opt.star.sed.sed.max()), opt.diagnostics)   
       
      opt.planet_sed_original = copy.deepcopy(opt.planet.sed.sed)
      # opt.planet.sed.sed  *= opt.star.sed.sed
      # opt.planet.sed.sed  *= opt.d_x_wav_osr
      
      jexosim_msg ("check 1.5:  %s"%(opt.star.sed.sed.max()), opt.diagnostics)                    
      opt.star.sed.sed     *= opt.d_x_wav_osr   
      opt.star.sed_it.sed     *= opt.d_x_wav_osr   
      jexosim_msg ("check 2:  %s"%(opt.star.sed.sed.max()), opt.diagnostics)  
      jexosim_plot('star sed check 2.0', opt.diagnostics, 
                   xdata=opt.x_wav_osr, ydata=opt.star.sed.sed , marker='-')

      
      return opt
      
    
      


       
