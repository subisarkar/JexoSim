"""
JexoSim 2.0

Signal library

"""
import numpy as np
from jexosim.lib.jexosim_lib import jexosim_msg, jexosim_plot
from jexosim.lib import jexosim_lib, noise_lib 
from astropy import units as u
import matplotlib.pyplot as plt
import sys

def fast_method(opt):
       
  idx0, idx1, idxA, idxB = crop_array(opt) 
  # apply x crop to 1D arrays    
  opt.x_wav_osr0 = opt.x_wav_osr*1
  opt.x_wav_osr = opt.x_wav_osr[idxA*3:idxB*3]
  opt.x_pix_osr = opt.x_pix_osr[idxA*3:idxB*3]
  opt.cr_wl = opt.cr_wl[idxA:idxB]
  opt.cr = opt.cr[idxA:idxB]
  if opt.timeline.apply_lc.val ==1:
       opt.ldc = opt.ldc[:,idxA:idxB]
       opt.lc  = opt.lc[idxA:idxB]   
  opt.zodi.sed = opt.zodi.sed[idxA*3:idxB*3]
  opt.sunshield.sed = opt.sunshield.sed[idxA*3:idxB*3]
  opt.emission.sed = opt.emission.sed[idxA*3:idxB*3]
  opt.quantum_yield.sed = opt.quantum_yield.sed[idxA*3:idxB*3]
  opt.qy_zodi  = opt.qy_zodi[idxA*3:idxB*3]
  opt.qy_sunshield  = opt.qy_sunshield[idxA*3:idxB*3]
  opt.qy_emission = opt.qy_emission[idxA*3:idxB*3]
  opt.exp_sig = opt.exp_sig[idxA:idxB] #expected final star signal in R bin from detector module sanity check
  # apply x and y crop to 2D arrays     
  opt.qe = opt.qe[idx0:idx1][:, idxA:idxB]
  opt.qe_uncert = opt.qe_uncert[idx0:idx1][:, idxA:idxB]
  opt.fp = opt.fp[idx0*3:idx1*3][:, idxA*3:idxB*3]
  opt.fp_signal = opt.fp_signal[idx0*3:idx1*3][:, idxA*3:idxB*3]

  # obtain proxy bkg signal for pipeline background subtraction step   
  opt.bkg_signal = proxy_bkg(opt)
  jexosim_msg ("fast method used, focal place image reduced from  %s x %s to %s x %s"%(opt.fp_original.shape[0]/3, opt.fp_original.shape[1]/3 , opt.fp.shape[0]/3, opt.fp.shape[1]/3), opt.diagnostics)

  return opt


def crop_array(opt):      
  fp_whole = opt.fp[1::3,1::3]
  fp_signal_whole = opt.fp_signal[1::3,1::3]
  #==============================================================================   
  #1) Calculate the maximum width (y crop) in pixels and apply crop
  #==============================================================================   
  aa = fp_signal_whole.sum(axis=1).value # sum of profile in x axis
  jexosim_plot('y crop selection', opt.diagnostics, ydata=aa, marker='ro-', grid=True)
  
  bb = np.mean(np.vstack((aa[:5],aa[-5:]))) # average of outer 5 pixels
  if bb == 0:
      aa_ =  aa[np.argwhere(aa>0).T[0]]
      bb = aa_.min()/np.e
  idx_max = np.argmax(aa)
  
  print (bb)
 
  #find where signal falls below b*npe either side
  for i in range(int(len(aa)/2)):
      s = aa[idx_max-i]
      if s < bb*np.e:
          # idx0=1+ idx_max-i #+1 so that it starts within the region of > b8npe
          idx0=idx_max-i #+1 so that it starts within the region of > b8npe
          break    
   
    
  for i in range(int(len(aa)/2)):
      s = aa[idx_max+i]
      if s < bb*np.e:
          # idx1=idx_max+i
          idx1=1+idx_max+i
          break        
  #idx0, idx1 #indexs that bound this region
  idx1 = int(idx1) ; idx0 = int(idx0)
  print (idx0, idx1)
  w_crop = idx1- (idx0)  
  jexosim_msg ("width of crop chosen %s"%(w_crop ) , opt.diagnostics)  
  jexosim_plot('y crop selection', opt.diagnostics, xdata=np.arange(idx0, idx1,1), ydata=aa[idx0:idx1], 
               marker='bo-', grid=True)
  #==============================================================================   
  #2) Calculate the maximum length (x crop) in pixels and apply crop
  #==============================================================================     
  wav_sol= opt.x_wav_osr[1::3].value # wav sol in whole pixels
  idx = np.argwhere((wav_sol>=opt.channel.pipeline_params.start_wav.val-0.5)& (wav_sol<=opt.channel.pipeline_params.end_wav.val+0.5)) 
  idxA = idx[0].item()
  idxB = idx[-1].item()
  
  return idx0, idx1, idxA, idxB


def proxy_bkg(opt):
#==============================================================================   
#Obtain a value for the background subtraction in pipeline as if the crop did not happened
#==============================================================================        
# a) set up background array for each ndr, of 10 pixels wide vs cropped fp in length 
  bkg_signal = np.zeros((10, opt.fp[1::3,1::3].shape[1], opt.n_ndr))*u.electron
  blank_bkg = np.ones((10, opt.fp[1::3,1::3].shape[1], opt.n_ndr))

  star_signal = np.zeros_like(blank_bkg)*u.electron
  zodi_signal = np.zeros_like(blank_bkg)*u.electron
  sunshield_signal = np.zeros_like(blank_bkg)*u.electron
  emission_signal = np.zeros_like(blank_bkg)*u.electron
  dc_signal = np.zeros_like(blank_bkg)*u.electron
  
  if opt.background.EnableZodi.val == 1:
      zodi_signal = blank_bkg* opt.zodi.sed[opt.offs::opt.osf][np.newaxis, :, np.newaxis]     
      zodi_signal = zodi_signal * opt.frame_time * opt.frames_per_ndr
      bkg_signal = bkg_signal + zodi_signal 
  if opt.background.EnableSunshield.val == 1:
      sunshield_signal = blank_bkg* opt.sunshield.sed[opt.offs::opt.osf][np.newaxis, :, np.newaxis]     
      sunshield_signal = sunshield_signal * opt.frame_time * opt.frames_per_ndr
      bkg_signal = bkg_signal + sunshield_signal 
      
  if opt.background.EnableEmission.val == 1:       
      emission_signal = blank_bkg* opt.emission.sed[opt.offs::opt.osf][np.newaxis, :, np.newaxis] 
      emission_signal = emission_signal * opt.frame_time * opt.frames_per_ndr
      bkg_signal = bkg_signal + emission_signal      
  if opt.noise.ApplyPRNU.val == 1:
      qe_bkg = np.vstack((opt.qe[0:5], opt.qe[-5:])) # pick QE at edges of grid, where background sampled
      qe_uncert_bkg = np.vstack((opt.qe_uncert[0:5], opt.qe_uncert[-5:]))
      applied_qe_bkg = qe_bkg*qe_uncert_bkg 
      bkg_signal= (bkg_signal.transpose() * applied_qe_bkg.transpose() ).transpose()           
  if opt.background.EnableDC.val ==1:
      dc_signal = blank_bkg*  opt.channel.detector_pixel.Idc() * opt.frame_time * opt.frames_per_ndr                 
      bkg_signal = bkg_signal + dc_signal       
  combined_noise  = np.zeros_like(bkg_signal) 
  if opt.noise.EnableShotNoise.val == 1:       
      bkg_signal = np.where(bkg_signal >= 0.0, bkg_signal, 0)
      combined_noise = noise_lib.calc_poission_noise(bkg_signal.value) *u.electron     
  # Apply weighted quantum yield to signal
  qy_emission = opt.qy_emission[1::3][np.newaxis, :, np.newaxis]
  qy_zodi = opt.qy_zodi[1::3][np.newaxis, :, np.newaxis]
  qy_sunshield = opt.qy_sunshield[1::3][np.newaxis, :, np.newaxis] 
  qy_star = opt.quantum_yield.sed[1::3][np.newaxis, :, np.newaxis]
  
  # # this works  
  # quantum_yield_total = (qy_zodi*zodi_signal +  qy_sunshield*sunshield_signal + qy_emission*emission_signal 
  #                        +1*dc_signal) / bkg_signal
  # bkg_signal = bkg_signal*quantum_yield_total
  # combined_noise = combined_noise*quantum_yield_total  
  
  # this also works # star signal is zero
  photons_total = star_signal + zodi_signal + sunshield_signal + emission_signal
  photons_total = np.where(photons_total<=0, 1e-10*u.electron, photons_total)
  quantum_yield_total = (qy_star*star_signal + qy_zodi*zodi_signal + qy_sunshield*sunshield_signal + qy_emission*emission_signal) / photons_total
  quantum_yield_total = np.where(quantum_yield_total <1, 1, quantum_yield_total)
  # print (quantum_yield_total.max(), quantum_yield_total.min())

  if quantum_yield_total.max() > 2 or  quantum_yield_total.min() <1:
      jexosim_msg('error in QY calculation for background')
      sys.exit()

  # now only apply to photons, so remove dc, multiply in qy and add back dc
  signal = (bkg_signal - dc_signal)*quantum_yield_total + dc_signal 
  noise  = (combined_noise - (dc_signal/bkg_signal)*combined_noise)*quantum_yield_total + (dc_signal/bkg_signal)*combined_noise
  bkg_signal = signal
  combined_noise  = noise
 
  # n= combined_noise.sum(axis=0)
  # n = n.std(axis=1)
  # plt.figure('proxy background noise pixel level')
  # plt.plot(opt.x_wav_osr[1::3], n, 'r-')
  
  # Obtain fano noise
  if opt.noise.EnableFanoNoise.val == 1:
      
      # old way
      # signal_for_fano = zodi_signal + sunshield_signal + emission_signal # use only photons for fano noise
      # signal_for_fano = np.where(signal_for_fano<=0, 1e-10*u.electron, signal_for_fano)
      # quantum_yield_for_fano = (qy_zodi*zodi_signal + qy_sunshield*sunshield_signal + qy_emission*emission_signal) / signal_for_fano
      # quantum_yield_for_fano = np.where(quantum_yield_for_fano<1, 1, quantum_yield_for_fano)
      # fano_noise = noise_lib.calc_fano_noise(quantum_yield_for_fano, signal_for_fano)
      
       # new way
      fano_noise = noise_lib.calc_fano_noise(quantum_yield_total, photons_total)
        
      # n= fano_noise.sum(axis=0)
      # n = n.std(axis=1)
      # plt.figure('proxy background combined noise pixel level')
      # plt.plot(opt.x_wav_osr[1::3], n, 'b-')
     
      combined_noise = combined_noise + fano_noise
  
      # n= combined_noise.sum(axis=0)
      # n = n.std(axis=1)
      # plt.plot(opt.x_wav_osr[1::3], n, 'b-')
      # xxx
          
  if opt.noise.EnableShotNoise.val == 1 or opt.noise.EnableFanoNoise.val == 1:
       if opt.simulation.sim_full_ramps.val == 0 and opt.simulation.sim_use_UTR_noise_correction.val == 1:
             n = opt.projected_multiaccum
             combined_noise, scale = noise_lib.poission_noise_UTR_correction(n, combined_noise.value) 
             combined_noise = combined_noise * u.electron
          
  bkg_signal = bkg_signal + combined_noise 

  for i in range(0, opt.n_ndr, opt.effective_multiaccum):
      bkg_signal[...,i:i+opt.effective_multiaccum] = np.cumsum(bkg_signal[...,i:i+opt.effective_multiaccum], axis=2)
  if opt.simulation.sim_use_ipc.val == 1 and opt.channel.instrument.val !='MIRI': 
      ipc_kernel = np.load(opt.channel.detector_array.ipc_kernel.val.replace('__path__', '%s/%s'%(opt.jexosim_path, 'jexosim')))
      bkg_signal =  noise_lib.apply_ipc(bkg_signal, ipc_kernel)
  if opt.noise.EnableReadoutNoise.val == 1:
      bkg_signal =  noise_lib.apply_read_noise(bkg_signal, opt)     
# =============================================================================
#       now do pipeline steps for bkg_signal  
# =============================================================================
# subtract the dc
  if opt.background.EnableDC.val ==1:  
      bkg_signal =  bkg_signal - opt.channel.detector_pixel.Idc.val* opt.duration_per_ndr
 # flat field (but with uncertainty)
  if opt.noise.ApplyPRNU.val == 1:
       jexosim_msg ("APPLYING Flat field to background counts..." , opt.diagnostics)
       bkg_signal =  (bkg_signal.transpose() / qe_bkg.transpose() ).transpose()

  return bkg_signal


def initiate_signal(opt):
  if opt.simulation.sim_use_fast.val ==1 and opt.channel.instrument.val!='NIRISS' and \
       opt.fp[1::3,1::3].shape[0]>20: #fast method not suitable for NIRISS due to curved spectrum or if the width is small
       opt = fast_method(opt) 
  opt.signal_only = noise_lib.obtain_signal_only(opt) 
 
  return opt

def initiate_noise(opt):
    opt.combined_noise  = np.zeros_like(opt.signal)
    opt.star_signal  = opt.signal*1 
    return opt

def apply_lc(opt):
    if opt.timeline.apply_lc.val ==0:
      jexosim_msg ("OMITTING LIGHT CURVE...", opt.diagnostics)
      return opt
    else:
      jexosim_msg ("APPLYING LIGHT CURVE...", opt.diagnostics)
      signal = opt.signal*1
      lc = opt.lc*1
      signal = signal*lc
      opt.signal = signal 
      return opt
    
def apply_systematic(opt):
    if opt.simulation.sim_use_systematic_model.val == 0: 
      return opt
    else:
      jexosim_msg ("Applying systematic model...", opt.diagnostics)
      signal = opt.signal*1
      syst = opt.syst*1
      signal = signal*syst
      opt.signal = signal 
      return opt  
   
def apply_non_stellar_photons(opt):
  blank_fp_shape = np.ones((int(opt.fp.shape[0]/3), int(opt.fp.shape[1]/3), len(opt.frames_per_ndr)))
  opt.zodi_signal = np.zeros_like(blank_fp_shape)*u.electron
  opt.sunshield_signal = np.zeros_like(blank_fp_shape)*u.electron
  opt.emission_signal = np.zeros_like(blank_fp_shape)*u.electron
  if opt.background.EnableZodi.val == 1:
      jexosim_msg ("APPLYING ZODI..." , opt.diagnostics)     
      zodi_signal = blank_fp_shape* opt.zodi.sed[opt.offs::opt.osf][np.newaxis, :, np.newaxis]      
      opt.zodi_signal = zodi_signal * opt.frame_time * opt.frames_per_ndr
      opt.signal = opt.signal + opt.zodi_signal 
  else:
      jexosim_msg ("NOT APPLYING ZODI..." , opt.diagnostics)  
      
  if opt.background.EnableSunshield.val == 1:
      jexosim_msg ("APPLYING SUNSHIELD SIGNAL..." , opt.diagnostics)     
      sunshield_signal = blank_fp_shape* opt.sunshield.sed[opt.offs::opt.osf][np.newaxis, :, np.newaxis]      
      opt.sunshield_signal = sunshield_signal * opt.frame_time * opt.frames_per_ndr
      opt.signal = opt.signal + opt.sunshield_signal 
  else:
      jexosim_msg ("NOT APPLYING SUNSHIELD SIGNAL..." , opt.diagnostics)       
        
  if opt.background.EnableEmission.val == 1:
      jexosim_msg ( "APPLYING EMISSION..."  ,  opt.diagnostics  )         
      emission_signal = blank_fp_shape* opt.emission.sed[opt.offs::opt.osf][np.newaxis, :, np.newaxis]       
      opt.emission_signal = emission_signal * opt.frame_time * opt.frames_per_ndr
      opt.signal = opt.signal + opt.emission_signal     
  else:
      jexosim_msg ("NOT APPLYING EMISSION..." , opt.diagnostics) 
  return opt

def apply_prnu(opt):
    opt.qe_grid = opt.qe # used for flat field in pipeline 
    jexosim_msg ("mean and standard deviation of flat field: should match applied flat in data reduction %s %s"%(opt.qe.mean(), opt.qe.std()),  opt.diagnostics  )
    jexosim_msg ("standard deviation of flat field uncertainty %s"%(opt.qe_uncert.std()), opt.diagnostics  )     
    applied_qe= opt.qe*opt.qe_uncert 
    if opt.noise.ApplyPRNU.val == 1: 
        jexosim_msg ("PRNU GRID BEING APPLIED", opt.diagnostics) 
        opt.signal= (opt.signal.transpose() * applied_qe.transpose() ).transpose()  
    else:
        jexosim_msg ("PRNU GRID NOT APPLIED...", opt.diagnostics) 
    return opt

def apply_dc(opt):
  blank_fp_shape = np.ones((int(opt.fp.shape[0]/3), int(opt.fp.shape[1]/3), len(opt.frames_per_ndr)))
  opt.dc_signal = np.zeros_like(blank_fp_shape)*u.electron
  if opt.background.EnableDC.val ==1:
      jexosim_msg ("DARK CURRENT being added...%s"%(opt.channel.detector_pixel.Idc.val ), opt.diagnostics  )
      opt.dc_signal = blank_fp_shape*  opt.channel.detector_pixel.Idc() * opt.frame_time * opt.frames_per_ndr                 
      opt.signal = opt.signal + opt.dc_signal     
  else:
      jexosim_msg ("DARK CURRENT.. not... being added...", opt.diagnostics)
  return opt
     
def apply_quantum_yield(opt):     
   # Apply weighted quantum yield to signal
   jexosim_msg ("Applying quantum yield", opt.diagnostics)
   qy_emission = opt.qy_emission[1::3][np.newaxis, :, np.newaxis]
   qy_zodi = opt.qy_zodi[1::3][np.newaxis, :, np.newaxis]
   qy_sunshield = opt.qy_sunshield[1::3][np.newaxis, :, np.newaxis]    
   qy_star = opt.quantum_yield.sed[1::3][np.newaxis, :, np.newaxis]
   
   print (qy_star.max(), qy_star.min())
   print (qy_zodi.max(), qy_zodi.min())
   print (qy_sunshield.max(), qy_sunshield.min())
   print (qy_emission.max(), qy_emission.min())
 
   print (opt.star_signal.max(), opt.star_signal.min())
   print (opt.zodi_signal.max(), opt.zodi_signal.min())
   print (opt.sunshield_signal.max(), opt.sunshield_signal.min())
   print (opt.emission_signal.max(), opt.emission_signal.min())
   
   

   ## This works
   # signal_total = opt.star_signal + opt.zodi_signal + opt.sunshield_signal + opt.emission_signal + opt.dc_signal
   # signal_total = np.where(signal_total<=0, 1e-10*u.electron, signal_total)
   # opt.quantum_yield_total = (qy_star*opt.star_signal + qy_zodi*opt.zodi_signal + qy_sunshield*opt.sunshield_signal + qy_emission*opt.emission_signal + 1*opt.dc_signal) / signal_total
   # opt.quantum_yield_total = np.where(opt.quantum_yield_total<1, 1, opt.quantum_yield_total)
   # opt.signal = opt.signal*opt.quantum_yield_total
   # opt.combined_noise = opt.combined_noise*opt.quantum_yield_total
   
   ## This also works - newer
   photons_total = opt.star_signal + opt.zodi_signal + opt.sunshield_signal + opt.emission_signal
   photons_total = np.where(photons_total<=0, 1e-10*u.electron, photons_total)
   opt.quantum_yield_total = (qy_star*opt.star_signal + qy_zodi*opt.zodi_signal + qy_sunshield*opt.sunshield_signal + qy_emission*opt.emission_signal) / photons_total
 
   opt.quantum_yield_total = np.where(opt.quantum_yield_total <1, 1, opt.quantum_yield_total)
 
   # now only apply to photons, so remove dc, multiply in qy and add back dc
   signal = (opt.signal - opt.dc_signal)*opt.quantum_yield_total + opt.dc_signal 
   noise  = (opt.combined_noise - (opt.dc_signal/opt.signal)*opt.combined_noise)*opt.quantum_yield_total + (opt.dc_signal/opt.signal)*opt.combined_noise
   noise[np.isnan(noise)] = 0 # mostly for read noise where opt.signal = zero returns nans
   opt.signal = signal
   opt.combined_noise  = noise
   
   if opt.quantum_yield_total.max() > 2 or  opt.quantum_yield_total.min() <1:
       jexosim_msg('error in QY calculation')
       sys.exit()
  
   jexosim_msg(f'max, min quantum yield applied: {opt.quantum_yield_total.max()}, {opt.quantum_yield_total.min()}', opt.diagnostics)

   return opt

def apply_combined_noise(opt):
    opt.signal = opt.signal + opt.combined_noise
    return opt

def apply_utr_correction(opt):
  # UTR correction verfied as correct for read noise, poisson noise, fano noise, fano and poission noise
  if opt.noise.EnableShotNoise.val == 1 or opt.noise.EnableFanoNoise.val == 1:
       if opt.simulation.sim_full_ramps.val == 0 and opt.simulation.sim_use_UTR_noise_correction.val == 1:
             jexosim_msg ("applying correction to photon noise for UTR read", opt.diagnostics  )
             n = opt.projected_multiaccum
             opt.combined_noise, scale = noise_lib.poission_noise_UTR_correction(n, opt.combined_noise.value)              
             opt.combined_noise = opt.combined_noise * u.electron
  return opt

def make_ramps(opt):
 
  jexosim_msg ("check point 6.1 %s"%(opt.signal.max()) , opt.diagnostics  )
  # Make ramps 
  for i in range(0, opt.n_ndr, opt.effective_multiaccum):
      opt.signal[...,i:i+opt.effective_multiaccum] = np.cumsum(opt.signal[...,i:i+opt.effective_multiaccum], axis=2) 
  return opt

def apply_ipc(opt):
  # Apply IPC: should be before read noise
  if opt.simulation.sim_use_ipc.val == 1 and opt.channel.instrument.val !='MIRI': 
       jexosim_msg('Applying IPC...', opt.diagnostics)    
       ipc_kernel = np.load(opt.channel.detector_array.ipc_kernel.val.replace('__path__', '%s/%s'%(opt.jexosim_path, 'jexosim')))
       opt.signal =  noise_lib.apply_ipc(opt.signal, ipc_kernel)
  return opt

def apply_read_noise(opt): 
  if opt.noise.EnableReadoutNoise.val == 1:
      jexosim_msg ("READ NOISE... being added...", opt.diagnostics)
      opt.signal = noise_lib.apply_read_noise(opt.signal,opt)         
  else:
      jexosim_msg ("READ NOISE...not... being added..." , opt.diagnostics  )                
  jexosim_msg ("check point 7 %s %s"%(opt.signal.max(), opt.signal.min()) , opt.diagnostics  )
  return opt
  
def apply_jitter(opt):   
   if opt.noise.EnableSpatialJitter.val  ==1 or opt.noise.EnableSpectralJitter.val==1:
       opt = noise_lib.simulate_jitter(opt)
   else:
       opt = noise_lib.jitterless(opt)
   if opt.noise.EnableSpatialJitter.val== 0:
        opt.pointing_timeline[:,2] =  opt.pointing_timeline[:,2]*0
   if opt.noise.EnableSpectralJitter.val== 0:
        opt.pointing_timeline[:,1] =  opt.pointing_timeline[:,1]*0   
   if opt.noise.DisableAll.val == 1:   
        opt.pointing_timeline[:,1] = opt.pointing_timeline[:,1]*0 
        opt.pointing_timeline[:,2] = opt.pointing_timeline[:,2]*0
   opt.signal[np.isnan(opt.signal)] = 0
   opt.signal = np.where(opt.signal >= 0.0, opt.signal, 0)  

   return opt 
       
       
def apply_fano(opt): 
    if opt.noise.EnableFanoNoise.val ==1:
        jexosim_msg('Applying Fano noise...', opt.diagnostics)
        # old...
        # consider photons only, not dc. 
        # qy_emission = opt.qy_emission[1::3][np.newaxis, :, np.newaxis]
        # qy_zodi = opt.qy_zodi[1::3][np.newaxis, :, np.newaxis] 
        # qy_sunshield = opt.qy_sunshield[1::3][np.newaxis, :, np.newaxis] 
        # qy_star = opt.quantum_yield.sed[1::3][np.newaxis, :, np.newaxis]
        # signal_total = opt.star_signal + opt.zodi_signal + +opt.sunshield_signal + opt.emission_signal  
        # signal_total = np.where(signal_total<=0, 1e-10*u.electron, signal_total)
        # quantum_yield_for_fano = (qy_star*opt.star_signal + qy_zodi*opt.zodi_signal + qy_sunshield*opt.sunshield_signal +  qy_emission*opt.emission_signal) / signal_total
        # quantum_yield_for_fano = np.where(quantum_yield_for_fano<1, 1, quantum_yield_for_fano)
        # jexosim_msg(f'max, min quantum yield applied for fano: {quantum_yield_for_fano.max()}, {quantum_yield_for_fano.min()}', opt.diagnostics)
        # jexosim_plot('quantum yield total', 1, xdata=opt.x_wav_osr[1::3], ydata=(quantum_yield_for_fano.mean(axis=2)).mean(axis=0) )
        # fano_noise = noise_lib.calc_fano_noise(quantum_yield_for_fano, signal_total)
               
        # new... same result
        photons_total = opt.star_signal + opt.zodi_signal + opt.sunshield_signal + opt.emission_signal
        photons_total = np.where(photons_total<=0, 1e-10*u.electron, photons_total)
        fano_noise = noise_lib.calc_fano_noise(opt.quantum_yield_total, photons_total)
        
        # plt.figure('fano noise pixel level')
        # n = fano_noise.sum(axis=0)
        # n = n.std(axis=1)
        # plt.plot(opt.x_wav_osr[1::3], n, 'g-')
                
        opt.combined_noise = opt.combined_noise + fano_noise
        
        # plt.figure('combined noise pixel level')
        # n = opt.combined_noise.sum(axis=0)
        # n = n.std(axis=1)
        # plt.plot(opt.x_wav_osr[1::3], n, 'b-')      
         
    else:
        jexosim_msg('Fano noise... not... applied', opt.diagnostics)
        
    return opt

def apply_poisson_noise(opt):    
    if opt.noise.EnableShotNoise.val == 1:    
       jexosim_msg ("Applying Poisson noise", opt.diagnostics)
       opt.signal = np.where(opt.signal >= 0.0, opt.signal, 0)
       opt.combined_noise = noise_lib.calc_poission_noise(opt.signal.value) *u.electron
       
    else:
       jexosim_msg ("Poisson noise not being applied...", opt.diagnostics)
    return opt

def apply_gaps(opt):   
#========Apply gaps to Hi res NIRSpec arrays=================================== 
  if opt.channel.name == 'NIRSpec_BOTS_G140H_F100LP' or \
      opt.channel.name == 'NIRSpec_BOTS_G235H_F170LP'\
        or opt.channel.name == 'NIRSpec_BOTS_G395H_F290LP':      
      wav = opt.x_wav_osr[1::3]*1
      if opt.diagnostics == 1:
          import matplotlib.pyplot as plt
          plt.figure('wav test ')
          plt.plot(opt.x_wav_osr[1::3])  
      idx0 = np.argwhere(wav==opt.wav_gap_start)[0].item() #recover the start of the gap
      idx1 = idx0 + opt.gap_len # gap of 172 pix works well for G1H and G2H, but a little off (by 0.01 microns for the other in some cases from published values)
      opt.signal[:,idx0:idx1,:] = 0
      if opt.pipeline.useSignal.val == 1:
           opt.signal_only[:,idx0:idx1,:] = 0 
      if opt.diagnostics ==1:     
          import matplotlib.pyplot as plt
          plt.figure('show gap in focal plane image on an NDR')
          plt.imshow(opt.signal[...,1].value,aspect='auto', interpolation = None)
          wl = opt.x_wav_osr.value[1::3]*1
          wl[idx0:idx1] = 0
          plt.figure('show gap in wl solution')
          plt.plot(wl, 'ro')
          plt.figure('compare to detector')
          plt.plot(wl, opt.signal[...,1].sum(axis=0))
  return opt
      
