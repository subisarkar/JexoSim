"""
JexoSim 
2.0
Noise module
v.1.0

"""

import numpy as np
from jexosim.lib.jexosim_lib import jexosim_msg, jexosim_plot
from jexosim.lib import jexosim_lib, noise_lib, signal_lib 
from astropy import units as u
import copy

def run(opt):
 
  opt.fp =   copy.deepcopy(opt.fp_original) # needed to reuse if cropped fp is used and monte carlo mode used
  opt.fp_signal =  copy.deepcopy(opt.fp_signal_original) # " 
  opt.zodi.sed =  copy.deepcopy(opt.zodi_sed_original) # "
  opt.emission.sed =  copy.deepcopy(opt.emission_sed_original) # "
  opt.lc = copy.deepcopy(opt.lc_original)    
  opt.ldc = copy.deepcopy(opt.ldc_original)
  opt.cr_wl = copy.deepcopy(opt.cr_wl_original) 
  opt.cr = copy.deepcopy(opt.cr_original)  
  opt.x_wav_osr = copy.deepcopy(opt.x_wav_osr_original)
  opt.x_pix_osr = copy.deepcopy(opt.x_pix_osr_original)
  opt.qe = copy.deepcopy(opt.qe_original)
  opt.qe_uncert = copy.deepcopy(opt.qe_uncert_original)
  opt.quantum_yield = copy.deepcopy(opt.quantum_yield_original)
  opt.qy_zodi = copy.deepcopy(opt.qy_zodi_original)
  opt.qy_emission = copy.deepcopy(opt.qy_emission_original)
  opt.syst_grid = copy.deepcopy(opt.syst_grid_original)
   
  opt = signal_lib.initiate_signal(opt) 
  opt = signal_lib.apply_jitter(opt)
  opt = signal_lib.apply_lc(opt)
  opt = signal_lib.initiate_noise(opt)
  opt = signal_lib.apply_non_stellar_photons(opt)
  opt = signal_lib.apply_prnu(opt)
  opt = signal_lib.apply_dc(opt)
  print ('www', opt.signal.max())
  
  opt = signal_lib.apply_poisson_noise(opt)
  
  print ('noise',opt.combined_noise.max()) 
  print ('After poisson', opt.signal.max())
  import matplotlib.pyplot as plt
  n= opt.combined_noise.sum(axis=0)
  n = n.std(axis=1)
  plt.figure('noise pixel level')
  plt.plot(opt.x_wav_osr[1::3], n, 'b-')
    
  opt = signal_lib.apply_quantum_yield(opt)
 
  
  print ('noise',opt.combined_noise.max())
  n= opt.combined_noise.sum(axis=0)
  n = n.std(axis=1)
  plt.figure('noise pixel level')
  plt.plot(opt.x_wav_osr[1::3], n, 'r-')
  
  opt = signal_lib.apply_fano(opt)
  
  n= opt.combined_noise.sum(axis=0)
  n = n.std(axis=1)
  plt.figure('noise pixel level')
  plt.plot(opt.x_wav_osr[1::3], n, 'g-')

  print ('noise',opt.combined_noise.max())

  print ('www', opt.signal.max())
  opt = signal_lib.apply_utr_correction(opt)
  
  plt.figure('noise pixel level')
  n= opt.combined_noise.sum(axis=0)
  n = n.std(axis=1)
  plt.plot(opt.x_wav_osr[1::3], n, 'r-')

  print ('www', opt.signal.max())
  opt = signal_lib.apply_combined_noise(opt)
  print ('www', opt.signal.max())
  
  opt = signal_lib.make_ramps(opt)
  opt = signal_lib.apply_ipc(opt)
  opt = signal_lib.apply_read_noise(opt)
  opt = signal_lib.apply_gaps(opt)
  opt.data = opt.signal
  opt.data_signal_only = opt.signal_only

  jexosim_plot('focal plane check1', opt.diagnostics, image=True, 
                  image_data=opt.fp_signal[1::3,1::3], aspect='auto', interpolation = None,
                  xlabel = 'x \'spectral\' pixel', ylabel = 'y \'spatial\' pixel')
  
  jexosim_plot('test - check NDR0', opt.diagnostics,
               image=True,  image_data = opt.data[...,0])
  jexosim_plot('test - check NDR1', opt.diagnostics,
               image=True,  image_data = opt.data[...,1])
  
  return opt
   
  
