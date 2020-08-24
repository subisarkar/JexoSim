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
import scipy.interpolate
import copy
 
def run(opt):
    
      jexosim_msg('channel check0 : %s'%(opt.star.sed.sed.max()), opt.diagnostics)
    
      ch = opt.channel
      
      tr_ =np.array([1.]*len(opt.common.common_wl))*u.dimensionless_unscaled
      
      ch.transmissions.optical_surface = ch.transmissions.optical_surface if isinstance(ch.transmissions.optical_surface, list) else \
                                                                        [ch.transmissions.optical_surface]
                                                                           
      for op in ch.transmissions.optical_surface:
          dtmp=np.loadtxt(op.transmission.replace('__path__', opt.__path__), delimiter=',')
          tr = Sed(dtmp[:,0]*u.um, dtmp[:,1]*u.dimensionless_unscaled)
          tr.rebin(opt.common.common_wl)
          tr_ *=tr.sed
            
      opt.channel_transmission = Sed(opt.common.common_wl, tr_) 
      opt.total_transmission =   Sed(opt.channel_transmission.wl, opt.telescope_transmission.sed*opt.channel_transmission.sed )
       
      jexosim_lib.sed_propagation(opt.star.sed, opt.channel_transmission)
      
#==============================================================================
# apply QE and convert to electrons
#==============================================================================
      dtmp=np.loadtxt(ch.detector_array.qe().replace('__path__', opt.__path__), delimiter=',')
      qe = Sed(dtmp[:,0]*u.um, dtmp[:,1]*u.dimensionless_unscaled)
      
      qe.rebin(opt.common.common_wl)

      PCE = Sed(opt.total_transmission.wl, opt.total_transmission.sed*qe.sed)
      TotTrans = Sed(opt.total_transmission.wl, opt.total_transmission.sed)

      jexosim_plot('PCE', opt.diagnostics, xdata = PCE.wl, ydata = PCE.sed)    
      jexosim_plot('Total transmission not including QE', opt.diagnostics, xdata = TotTrans.wl, ydata = TotTrans.sed)

      
      opt.qe_spec = qe
      opt.Re = qe.sed * (qe.wl).to(u.m)/(const.c.value * const.h.value * u.m)
      opt.star.sed.sed *= opt.Re*u.electron/u.W/u.s
      
      jexosim_msg('check 1.3 - Star sed max:  %s'%(opt.star.sed.sed.max()), opt.diagnostics)  
           
    ### create focal plane 
    #1# allocate focal plane with pixel oversampling such that Nyquist sampling is done correctly 
      fpn = ch.detector_array.array_geometry.val.split(',')
      fpn = [int(fpn[0]), int(fpn[1])]  
      opt.fpn = fpn    
      opt.fp  = np.zeros(( int(opt.fpn[0]*ch.simulation_factors.osf.val),
                           int(opt.fpn[1]*ch.simulation_factors.osf.val) ))
       
    #2# This is the current sampling interval in the focal plane.      
      opt.fp_delta = ch.detector_pixel.pixel_size() / ch.simulation_factors.osf()      
      opt.osf         = np.int(ch.simulation_factors.osf())
      opt.offs        = np.int(ch.simulation_factors.pix_offs())
   
#      opt.x_wav_osr, opt.x_pix_osr, opt.y_pos_osr = usePoly(opt)
      opt.x_wav_osr, opt.x_pix_osr, opt.y_pos_osr = useInterp(opt) 
      
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
      
      opt.star.sed.rebin(opt.x_wav_osr)         
      opt.planet.sed.rebin(opt.x_wav_osr)   

       
      jexosim_plot('planet sed', opt.diagnostics, xdata=opt.planet.sed.wl, ydata=opt.planet.sed.sed)

      opt.planet_sed_original = copy.deepcopy(opt.planet.sed.sed)
      opt.planet.sed.sed  *= opt.star.sed.sed
      opt.planet.sed.sed  *= opt.d_x_wav_osr

      jexosim_msg ("check 1.5:  %s"%(opt.star.sed.sed.max()), opt.diagnostics)  
                     
      opt.star.sed.sed     *= opt.d_x_wav_osr
          
      jexosim_msg ("check 2:  %s"%(opt.star.sed.sed.max()), opt.diagnostics)  

      jexosim_plot('star sed check 2.0', opt.diagnostics, 
                   xdata=opt.x_wav_osr, ydata=opt.star.sed.sed , marker='-')
      
      return opt
      
    
      

def useInterp(opt):
    
      offset = opt.fpn[1]*opt.channel.detector_pixel.pixel_size()/2.0  
        
      dtmp=np.loadtxt(opt.channel.camera.dispersion.path.replace(
	  '__path__', opt.__path__), delimiter=',')          
      ld = scipy.interpolate.interp1d(dtmp[...,2]*u.um + offset.to(u.um), dtmp[...,0], bounds_error=False,kind='slinear',
					fill_value=0.0)
          
      x_pix_osr = np.arange(opt.fp.shape[1]) * opt.fp_delta +  opt.fp_delta/2.0
      x_wav_osr = ld(x_pix_osr.to(u.um))*u.um # walength on each x pixel 
      
           
      pos_y = dtmp[...,1]  
       
      x_edge = np.arange(opt.fpn[1]+1) * opt.fp_delta*3
      opt.x_wav_edges = ld(x_edge.to(u.um))*u.um

      
      if pos_y[0] !=0:
          y_pos_osr = scipy.interpolate.interp1d(dtmp[...,0],  dtmp[...,1], bounds_error=False, fill_value=0.0)(x_wav_osr)
          # there must be a fill value as each x_wav_osr must have a corresponding y_pos_osr
          y_pos_osr = (y_pos_osr/((opt.fp_delta).to(u.um))).value

      else:
          y_pos_osr = []     
         
      return  x_wav_osr, x_pix_osr.to(u.um), y_pos_osr
      

def usePoly(opt):
    
      opt.channel.camera.dispersion.val = opt.fpn[1]*opt.channel.detector_pixel.pixel_size()/2.0  
       
      dtmp=np.loadtxt(opt.channel.camera.dispersion.path.replace(
	  '__path__', opt.__path__), delimiter=',')   
      pos = dtmp[...,2]*u.um + (opt.channel.camera.dispersion() ).to(u.um)
      wav = dtmp[...,0]
      pos_y = dtmp[...,1] 
      r = 7 # degree 7 works well
      z = np.polyfit(pos, wav, r)
      if pos_y[0] !=0:
          r_y = 7 # degree 7 works well
          z_y = np.polyfit(wav, pos_y, r_y)
 
      x_pix_osr = (np.arange(opt.fp.shape[1])*opt.fp_delta + opt.fp_delta/2).to(u.um)
      x_wav_osr = 0
      for i in range (0,r+1):
          x_wav_osr = x_wav_osr + z[i]*x_pix_osr.value**(r-i) 
      x_wav_osr =x_wav_osr*u.um
            
      idx = np.argwhere(x_wav_osr<wav.min())
      x_wav_osr[idx] = 0*u.um 
      idx = np.argwhere(x_wav_osr>wav.max())
      x_wav_osr[idx] = 0*u.um       
         
      idx = np.argwhere(x_pix_osr.value < pos.min())
      x_wav_osr[idx] = 0*u.um
      idx = np.argwhere(x_pix_osr.value > pos.max())
      x_wav_osr[idx] = 0*u.um               
     
      if pos_y[0] !=0:
          y_pos_osr =0
          for i in range (0,r_y+1):
              y_pos_osr = y_pos_osr + z_y[i]*x_wav_osr.value**(r_y-i) 
        
          y_pos_osr = (y_pos_osr/((opt.fp_delta).to(u.um))).value
      else:
          y_pos_osr = []
          
      return x_wav_osr, x_pix_osr, y_pos_osr
            
       
