"""
JexoSim 
2.0
Telescope module
v1.0

"""

from jexosim.classes.sed import Sed
from jexosim.lib import jexosim_lib
from jexosim.lib.jexosim_lib import jexosim_msg, jexosim_plot
import numpy           as np
from astropy import units as u
import copy 
import scipy
 
def run(opt):
    
      opt.osf         = np.int(opt.channel.simulation_factors.osf())
      opt.offs        = np.int(opt.channel.simulation_factors.pix_offs()) 
      
      fpn = opt.channel.detector_array.array_geometry.val.split(',')
      opt.fpn = [int(fpn[0]), int(fpn[1])]  

      opt.fp  = np.zeros(( int(opt.fpn[0]*opt.channel.simulation_factors.osf.val),
                            int(opt.fpn[1]*opt.channel.simulation_factors.osf.val) ))      
      opt.fp_delta = opt.channel.detector_pixel.pixel_size.val / opt.channel.simulation_factors.osf.val  
                     
#      opt.x_wav_osr, opt.x_pix_osr, opt.y_pos_osr = usePoly(opt)
      opt.x_wav_osr, opt.x_pix_osr, opt.y_pos_osr = useInterp(opt)  

      jexosim_msg('tel check 1: %s'%(opt.star.sed.sed.max()), opt.diagnostics)
      opt.star_sed = copy.deepcopy(opt.star.sed) #copy of star flux at telescope
         
      opt.Aeff = 0.25*np.pi*opt.common_optics.telescope_effective_diameter()**2
      opt.star.sed.sed*= opt.Aeff
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
      opt.planet.sed.rebin(opt.x_wav_osr) 
 
      jexosim_lib.sed_propagation(opt.star.sed, opt.telescope_transmission)
      jexosim_msg('tel check 3: %s'%(opt.star.sed.sed.max()), opt.diagnostics)
      return opt
  
def useInterp(opt):
    
      offset = opt.fpn[1]*opt.channel.detector_pixel.pixel_size()/2.0
 
      if opt.channel.name == 'NIRSpec_BOTS_G140H_F100LP' or opt.channel.name == 'NIRSpec_BOTS_G235H_F170LP'\
          or opt.channel.name == 'NIRSpec_BOTS_G395H_F290LP':
          offset = 1024*opt.channel.detector_pixel.pixel_size.val #centre the designated central wavelength at the boundary between subarray1024A and 1024b on NRS1
     
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
      

def usePoly(opt): #this needs updating
    
      offset = opt.fpn[1]*opt.channel.detector_pixel.pixel_size()/2.0 
      
      if opt.channel.name == 'NIRSpec_BOTS_G140H_F100LP' or opt.channel.name == 'NIRSpec_BOTS_G235H_F170LP'\
          or opt.channel.name == 'NIRSpec_BOTS_G395H_F290LP':
          offset = 1024*opt.channel.detector_pixel.pixel_size.val #centre the designated central wavelength at the boundary between subarray1024A and 1024b on NRS1
     
      dtmp=np.loadtxt(opt.channel.camera.dispersion.path.replace(
	  '__path__', opt.__path__), delimiter=',')  
      
      dtmp=np.loadtxt(opt.channel.camera.dispersion.path.replace(
	  '__path__', opt.__path__), delimiter=',')   
      pos = (dtmp[...,2]*u.um  + offset.to(u.um)) + (opt.channel.camera.dispersion() ).to(u.um)
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
            
  