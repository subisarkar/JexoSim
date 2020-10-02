"""
JexoSim 
2.0
Noise module
v.1.0

"""

import numpy as np
from jexosim.lib.jexosim_lib import jexosim_msg, jexosim_plot
from jexosim.lib import jexosim_lib 
from astropy import units as u
import copy
import time
import sys
from numba import jit, prange

def poission_noise (noise):    
    noise = np.random.poisson(noise)    
    return noise

def poission_noise_UTR_correction (n, noise):    
    noise_plus_pn =  np.random.poisson(noise)
    alpha = 6.0*(n**2+1)/(5.0*n*(n+1))
    noise_only = noise_plus_pn - noise
    # correction for utr: pn increased 
    noise_only = noise_only*np.sqrt(alpha) #scale the noise
    noise = noise + noise_only # add back noise to signal    
    return noise

def read_noise (noise, sigma_ro):    
    noise = noise + np.random.normal(scale = sigma_ro, size = noise.shape)  
    return noise

def read_noise_UTR_correction (n, noise, sigma_ro):   
    # correction for utr noise: achieves same noise ar utr but using cds      
    sigma_ro_prime = np.sqrt(12.0*(n-1)/(n*(n+1)))*sigma_ro /np.sqrt(2)  
    noise = noise + np.random.normal(scale = sigma_ro_prime, size = noise.shape)
    return noise     
 
@jit(nopython=True) 
def create_jitter_noise(fp, osf, frames_per_ndr, frame_osf, jitter_x, jitter_y):		
    n_ndr = len(frames_per_ndr)
    fp_whole = fp[int(osf/2.)::osf, int(osf/2.)::osf]
    ndr_count = np.zeros((fp_whole.shape[0], fp_whole.shape[1], n_ndr)) 
    ct = 0
    prog0 = 0        
    for j in range(n_ndr):
        prog =  float(j)/float(n_ndr) 
        a =int(prog*10)  
        if a>prog0:
            print (int(prog*100), '% complete')
        prog0=int(prog*10)

        count = np.zeros_like(fp[int(osf/2.)::osf, int(osf/2.)::osf])   
        
        print ('ccc', count.size)
        for i in range(frames_per_ndr[j]): # frames per ndr
            for k in range(frame_osf): # jitter subframes per frame
      
                off_x = jitter_x[ct]
                off_y = jitter_y[ct]
                count0 = fp[int(osf/2.)+off_y::osf, int(osf/2.)+off_x::osf]
                print ('ccc0', count0.size)
                count += count0      
                ct=ct+1                
        ndr_count[...,j] = count /  (frames_per_ndr[j]*frame_osf)         
    return ndr_count
  
@jit(nopython=True, parallel=True)
def jitter(fp0,  osf, frames_per_ndr, frame_osf, jitter_x , jitter_y, start_list, buffer, fp):
    
    buffer_offset = buffer*osf

    n_ndr = len(frames_per_ndr)
    fp_whole = fp[int(osf/2.)::osf, int(osf/2.)::osf]
    ndr_count = np.zeros((fp_whole.shape[0], fp_whole.shape[1], n_ndr)) 
   
    for j in prange(n_ndr):

        count = np.zeros_like(fp[int(osf/2.)::osf, int(osf/2.)::osf])   
        for i in prange(frames_per_ndr[j]): # frames per ndr
            for k in prange(frame_osf): # jitter subframes per frame                                              
                start = start_list[j] 
                ct = start + i*frame_osf + k # for accurate parallelisation ct must be defined in terms of j, k and i
                off_x = jitter_x[ct] 
                off_y = jitter_y[ct] 
                start_x = buffer_offset + int(osf/2.)+ off_x 
                start_y = buffer_offset + int(osf/2.)+ off_y
                end_x = start_x + fp_whole.shape[1]*osf 
                end_y = start_y + fp_whole.shape[0]*osf 
                
                count0 = fp0[start_y: end_y: osf, start_x: end_x : osf]
                count += count0      
                ct=ct+1                
        ndr_count[...,j] = count /  (frames_per_ndr[j]*frame_osf)                

    return ndr_count

 
def jitter2(fp0,  osf, frames_per_ndr, frame_osf, jitter_x , jitter_y, start_list, buffer, fp):
      
    buffer_offset = buffer*osf

    n_ndr = len(frames_per_ndr)
    fp_whole = fp[int(osf/2.)::osf, int(osf/2.)::osf]
    ndr_count = np.zeros((fp_whole.shape[0], fp_whole.shape[1], n_ndr)) 
   
    for j in range(n_ndr):

        count = np.zeros_like(fp[int(osf/2.)::osf, int(osf/2.)::osf])   
        for i in range(frames_per_ndr[j]): # frames per ndr
            for k in range(frame_osf): # jitter subframes per frame                                              
                start = start_list[j] 
                ct = start + i*frame_osf + k # for accurate parallelisation ct must be defined in terms of j, k and i
                off_x = jitter_x[ct] 
                off_y = jitter_y[ct] 
                start_x = buffer_offset + int(osf/2.)+ off_x 
                start_y = buffer_offset + int(osf/2.)+ off_y
                
                end_x = start_x + fp_whole.shape[1]*osf 
                end_y = start_y + fp_whole.shape[0]*osf 
                
                count0 = fp0[start_y: end_y: osf, start_x: end_x : osf]
                # print (off_x, off_y, count0.shape)
                count += count0      
                ct=ct+1                
        ndr_count[...,j] = count /  (frames_per_ndr[j]*frame_osf)                

    return ndr_count


def simulate_jitter(opt):
    
  jitter_x = opt.osf*(opt.yaw_jitter_effective/opt.channel.detector_pixel.plate_scale_x())
  jitter_y = opt.osf*(opt.pitch_jitter_effective/opt.channel.detector_pixel.plate_scale_y())
   
  jexosim_msg ("RMS jitter in pixel units x axis %s"%(np.std(jitter_x)/opt.osf), opt.diagnostics)
  jexosim_msg ("RMS jitter in pixel units y axis %s"%(np.std(jitter_y)/opt.osf), opt.diagnostics)

  fp_units = opt.fp.unit
  fp  	   = opt.fp.value
  osf      = np.int32(opt.osf)
  offs     = np.int32(opt.offs)
    
  magnification_factor = np.ceil( max(3.0/jitter_x.std(), 3.0/jitter_y.std()) )
           
  if (magnification_factor > 1):
        try:
          mag = np.int(magnification_factor.item()) | 1
        except:
          mag = np.int(magnification_factor) | 1        
        
  jexosim_msg ("mag %s"%(mag) , opt.diagnostics) 
  fp = jexosim_lib.oversample(fp, mag)
  osf *= mag
  offs = mag*offs + mag//2
  jitter_x *= mag
  jitter_y *= mag
  
  jexosim_msg ("spatial osf %s"%(osf) , opt.diagnostics) 
    
  if opt.noise.EnableSpatialJitter.val ==0 or opt.noise.DisableAll.val == 1: 
          jitter_y *= 0.0
  if opt.noise.EnableSpectralJitter.val ==0 or opt.noise.DisableAll.val == 1:
          jitter_x *= 0.0
    
  jitter_x = np.round(jitter_x)
  jitter_y = np.round(jitter_y) 
    
  #important for speed of jitter module that these not have units
  jitter_x = jitter_x.value
  jitter_y = jitter_y.value
    
  jexosim_msg('Applying jitter', 1)
     
  n_ndr = len(opt.frames_per_ndr)
  start_list =[]
  for j in range(n_ndr):
        if j ==0:
            start_list.append(0)
        else:            
            start_list.append(np.cumsum(opt.frames_per_ndr[0:j])[-1]*opt.frame_osf)
  start_list = np.array(start_list)

  aa = time.time()
  
  # cond =0
  # if int(fp.shape[0]/osf)*int(fp.shape[1]/osf) > 30000:
  #     print (int(fp.shape[0]/osf)*int(fp.shape[1]/osf))
  #     cond = 1 # splits array for jitter if above a certain size
 
  cond =1 # default to splitting as improved speed even for smaller arrays
  
  # edge buffer in whole pixels: needed to protect from errors arising when where jitter offset > half a pixel causing sampling to fall outside image area
  if jitter_y.max() > jitter_x.max():        
      buffer = int(np.ceil(jitter_y.max()/osf))    
  else:
      buffer = int(np.ceil(jitter_x.max()/osf)) 
  
  if cond == 0: # no spitting                   
      fp0 = np.zeros((fp.shape[0] + 2* buffer*osf, fp.shape[1] + 2* buffer*osf))
      fp0[buffer*osf: buffer*osf + fp.shape[0], buffer*osf: buffer*osf + fp.shape[1] ] = fp
                            
      noise =   jitter(fp0.astype(np.float32), 
        		 		osf.astype(np.int32),
        		 		opt.frames_per_ndr.astype(np.int32), 
        		 		opt.frame_osf.astype(np.int32),
        		 		jitter_x.astype(np.int32), 
        		 		jitter_y.astype(np.int32),
                        start_list.astype(np.int32),
                        buffer, 
                        fp.astype(np.float32)
                        ).astype(np.float32)  
 
       # noise =   create_jitter_noise(fp.astype(np.float32), 
       #     				osf.astype(np.int32),
       #     				opt.frames_per_ndr.astype(np.int32), 
       #     				opt.frame_osf.astype(np.int32),
       #     				jitter_x.astype(np.int32), 
       #     				jitter_y.astype(np.int32)).astype(np.float32)   
      
  if cond == 1: 
      jexosim_msg('Splitting array for jitter code', 1)

      overlap = 5*osf
      x_size= int(5000/ (fp.shape[0]/osf) ) *osf
  
      n_div = int(fp.shape[1]/x_size)
      jexosim_msg(f'Number of divisions {n_div}', opt.diagnostics)
      ct = 0
      
      # jexosim_msg(f'x_size {x_size} osf {osf} fp shape {fp.shape}', opt.diagnostics)
      
      for i in range(n_div): 
          bb = time.time()
          
          ct = i*x_size
          
          if i == 0 and i == n_div-1:
             x0 = 0
             x1 = None
          elif i == 0:        
             x0 = ct
             x1 = ct + x_size + overlap
          elif i == n_div-1:
             x0 = ct - overlap
             x1 = None     
          else:
             x0 = ct - overlap
             x1 = ct + x_size + overlap
             
          # jexosim_msg(f'x0 {x0} x1 {x1}', opt.diagnostics)

          div_fp= fp[:, x0:x1]
          
          # jexosim_msg(f'div_fp shape {div_fp.shape}', opt.diagnostics)
          
          div_fp0 = np.zeros((div_fp.shape[0] + 2* buffer*osf, div_fp.shape[1] + 2* buffer*osf))
          div_fp0[buffer*osf: buffer*osf + div_fp.shape[0], buffer*osf: buffer*osf + div_fp.shape[1] ] = div_fp

          div_noise =   jitter(div_fp0.astype(np.float32), 
        		 		osf.astype(np.int32),
        		 		opt.frames_per_ndr.astype(np.int32), 
        		 		opt.frame_osf.astype(np.int32),
        		 		jitter_x.astype(np.int32), 
        		 		jitter_y.astype(np.int32),
                        start_list.astype(np.int32),
                        buffer, 
                        div_fp.astype(np.float32)
                        ).astype(np.float32)  
 
          if i == 0 and i == n_div-1:
              div_noise_stack = div_noise  
          elif i==0:
              div_noise = div_noise[:, 0:-int(overlap/osf), : ]
              div_noise_stack = div_noise
          elif i == n_div-1:
              div_noise = div_noise[:, int(overlap/osf): , : ]
              div_noise_stack = np.hstack((div_noise_stack,div_noise)) 
          else:
              div_noise = div_noise[:, int(overlap/osf):-int(overlap/osf), : ]
              div_noise_stack = np.hstack((div_noise_stack,div_noise))  
         
          # jexosim_msg(f'div_noise_stack shape {div_noise_stack.shape}', opt.diagnostics)
          
          cc = time.time() -bb
          print(f'time remaining to complete jitter code {np.round((cc*(n_div-i)),2)} seconds', flush=True)
 
      jexosim_msg(f'noise stack shape after split jitter code {div_noise_stack.shape}', opt.diagnostics)  
      noise  = div_noise_stack
                  
  qq = opt.frames_per_ndr* fp_units*opt.frame_time
  noise = noise*qq 

  jexosim_msg('Time to run jitter code %s'%(time.time() - aa), opt.diagnostics) 
  
  return  noise 

#==============================================================================
#  Generates pointing timeline
#==============================================================================

def create_pointing_timeline(opt):
    # frames_per_ndr gives the number of frames in each ndr in sequence
    # ndr_end_frame_number gives the number of frames that have passed by the end of each NDR: take away the frames per ndr from ndr sequence to give a start index for the ndr in terms of frames
    # maccum = int(opt.effective_multiaccum)
 
    total_frames = int(np.sum(opt.frames_per_ndr))
    n_ndr = len(opt.frames_per_ndr)
    pointingArray = np.zeros((total_frames*opt.frame_osf, 3))    
    ct =0
    # for i in range (0, maccum*opt.n_exp, 1 ):
    for i in range (0, n_ndr, 1 ):
     
        idx = ct
        idx2 = int(idx + opt.frames_per_ndr[i]*opt.frame_osf)
        ct = idx2
    
        pointingArray[:,0][idx:idx2] = i        
        #This bit is very important for accuracy of jitter timeline > skips frames that fall in reset time
        start = int((opt.ndr_end_frame_number[i]- opt.frames_per_ndr[i])*opt.frame_osf)
        end =  int(start + opt.frames_per_ndr[i]*opt.frame_osf ) 
 
        pointingArray[:,1][idx:idx2] = opt.yaw_jitter[start:end]
        pointingArray[:,2][idx:idx2] = opt.pitch_jitter[start:end]
   
    return  pointingArray

 
   
#==============================================================================
# Generates noisy image stack and returns pointing timeline using a cropped array  
#==============================================================================   
def fast_method(opt):
       
  fp_whole = opt.fp[1::3,1::3]
  fp_whole0 = copy.deepcopy(fp_whole)
  qe = opt.qe 
  qe_uncert =  opt.qe_uncert 
  
  fp_signal_whole = opt.fp_signal[1::3,1::3]
  #==============================================================================   
  #1) Calculate the maximum width (y crop) in pixels and apply crop
  #==============================================================================   
  aa = fp_signal_whole.sum(axis=1).value # sum of profile in x axis
  
  jexosim_plot('y crop selection', opt.diagnostics, ydata=aa, marker='ro-', grid=True)
  
  bb = np.mean(np.vstack((aa[:5],aa[-5:]))) # average of outer 5 pixels
  idx_max = np.argmax(aa)
  #find where signal falls below b*npe either side
  for i in range(int(len(aa)/2)):
      s = aa[idx_max-i]
      if s < bb*np.e:
          idx0=1+ idx_max-i #+1 so that it starts within the region of > b8npe
          break
  for i in range(int(len(aa)/2)):
      s = aa[idx_max+i]
      if s < bb*np.e:
          idx1=idx_max+i
          break      
  
  #idx0, idx1 #indexs that bound this region
  idx1 = int(idx1) ; idx0 = int(idx0)
  w_crop = idx1- (idx0)  
  jexosim_msg ("width of crop chosen %s"%(w_crop ) , opt.diagnostics)
  
  jexosim_plot('y crop selection', opt.diagnostics, xdata=np.arange(idx0, idx1,1), ydata=aa[idx0:idx1], 
               marker='bo-', grid=True)

  # crop fp_whole and opt.fp in y 
  fp_whole= fp_whole[idx0:idx1]
  opt.fp = opt.fp[idx0*3:idx1*3]
  #==============================================================================   
  #2) Calculate the maximum length (x crop) in pixels and apply crop
  #==============================================================================     
  wav_sol= opt.x_wav_osr[1::3].value # wav sol in whole pixels
 
  idx = np.argwhere((wav_sol>=opt.channel.pipeline_params.start_wav.val-0.5)& (wav_sol<=opt.channel.pipeline_params.end_wav.val+0.5))
  idx = [idx[0].item(), idx[-1].item()] 

  #apply x crop to various arrays that need it:   
  opt.x_wav_osr = opt.x_wav_osr[idx[0]*3:idx[-1]*3]
  opt.x_pix_osr = opt.x_pix_osr[idx[0]*3:idx[-1]*3]
  opt.cr_wl = opt.cr_wl[idx[0]:idx[-1]]
  opt.cr = opt.cr[idx[0]:idx[-1]]
  if opt.timeline.apply_lc.val ==1:
      opt.ldc = opt.ldc[:,idx[0]:idx[-1]]
      opt.lc  = opt.lc[idx[0]:idx[-1]]      

  #apply x crop to fp_whole and opt.fp:            
  fp_whole = fp_whole[:, idx[0]:idx[-1]]
  opt.fp = opt.fp[:, idx[0]*3:idx[-1]*3]

   #==============================================================================   
   #3) Obtain a value for the background subtraction in pipeline as if the crop did not happened
   #==============================================================================     
#   1) set up background array for each ndr, of 10 pixels wide vs original fp in length and add zodi and emission
  bkg = np.zeros((10, fp_whole0.shape[1], opt.n_ndr))
  
  if (opt.background.EnableZodi.val == 1 or opt.background.EnableAll.val == 1) and opt.background.DisableAll.val != 1:
      bkg += (opt.zodi.sed[1::3].reshape(-1,1) * opt.frame_time * opt.frames_per_ndr).value 
  if (opt.background.EnableEmission.val == 1 or opt.background.EnableAll.val == 1) and opt.background.DisableAll.val != 1:
      bkg += (opt.emission.sed[1::3].reshape(-1,1) * opt.frame_time * opt.frames_per_ndr).value     
   #  2) apply QE to this diffuse background from same part of QE grid as the background
  
  qe_bkg = np.vstack((qe[0:5], qe[-5:])) # pick QE at edges of grid, where background sampled
  qe_uncert_bkg = np.vstack((qe_uncert[0:5], qe_uncert[-5:]))
  jexosim_msg ("QE std applied to background counts %s"%((qe_bkg*qe_uncert_bkg).std()) , opt.diagnostics)
  if (opt.noise.ApplyPRNU.val == 1 or opt.noise.EnableAll.val == 1) and opt.noise.DisableAll.val != 1:
       bkg =  np.rollaxis(bkg,2,0)
       bkg = bkg*qe_bkg*qe_uncert_bkg
       bkg =  np.rollaxis(bkg,0,3)       
         
 #  2a) apply DC to this background           
  if (opt.background.EnableDC.val ==1 or opt.background.EnableAll.val) and opt.background.DisableAll.val != 1:
      bkg += (opt.channel.detector_pixel.Idc() * opt.frame_time * opt.frames_per_ndr).value
 
 #  3) apply photon noise to the bkg
  bkg = np.where(bkg >= 0.0, bkg, 0) 
  if (opt.noise.EnableShotNoise.val == 1 or opt.noise.EnableAll.val == 1) and opt.noise.DisableAll.val != 1:
       bkg = np.random.poisson(bkg)
       
 #  4) make ramps      
  for i in range(0, opt.n_ndr, opt.effective_multiaccum):
      bkg[...,i:i+opt.effective_multiaccum] = np.cumsum(bkg[...,i:i+opt.effective_multiaccum], axis=2)
      
 #  4) flat field (but with uncertainty)
  if (opt.noise.ApplyPRNU.val == 1 or opt.noise.EnableAll.val == 1) and opt.noise.DisableAll.val != 1:
       jexosim_msg ("APPLYING Flat field to background counts..." , opt.diagnostics)
       bkg =  np.rollaxis(bkg,2,0)
       bkg = bkg/qe_bkg
       bkg =  np.rollaxis(bkg,0,3)  
       
#  5) crop bkg in x
       bkg = bkg[:,idx[0]:idx[-1],:]
  
  opt.bkg = bkg*u.electron
  
  #==============================================================================
  #4) crop diffuse background spectra in x for applying to main image later 
  #==============================================================================
  opt.zodi.sed = opt.zodi.sed[idx[0]*3:idx[-1]*3]
  opt.emission.sed = opt.emission.sed[idx[0]*3:idx[-1]*3]

  #============================================================================== 
  #5) crop qe grid in x and y  for later application
  #============================================================================== 
  qe = qe[idx0:idx1]
  qe = qe[:, idx[0]:idx[-1]]
  qe_uncert = qe_uncert[idx0:idx1]
  qe_uncert = qe_uncert[:, idx[0]:idx[-1]]
  opt.qe = qe
  opt.qe_uncert = qe_uncert  

  #============================================================================== 
  #6) crop signal only oversampled fp in x and y 
  #============================================================================== 
  opt.fp_signal = opt.fp_signal[idx0*3:idx1*3]
  opt.fp_signal = opt.fp_signal[:, idx[0]*3:idx[-1]*3]
  
  #(crop sanity check indicator) 
  opt.exp_sig = opt.exp_sig[idx[0]:idx[-1]] #expected final star signal in R bin from detector module sanity check
  
  jexosim_msg ("fast method used, focal place image reduced from  %s x %s to %s x %s"%(fp_whole0.shape[0], fp_whole0.shape[1] , fp_whole.shape[0], fp_whole.shape[1]), opt.diagnostics)
 
  return opt
        

def noise_simulator(opt): 

#==============================================================================
# Generates noisy image stack and returns pointing timeline   
#============================================================================== 

    #==============================================================================
    # set up jitterless or jittered pathways
    #==============================================================================

  if opt.pipeline.use_fast.val ==1 and opt.channel.instrument.val!='NIRISS' and \
       opt.fp[1::3,1::3].shape[0]>20:

  #fast method not suitable for NIRISS due to curved spectrum or if the width is small
      opt = fast_method(opt)
      fp = opt.fp[1::3,1::3]
  else:
      fp = opt.fp[1::3,1::3] 
      
  if (opt.noise.EnableSpatialJitter.val  ==1 or opt.noise.EnableSpectralJitter.val  ==1 or opt.noise.EnableAll.val == 1) and opt.noise.DisableAll.val != 1:
          jexosim_msg ("using jittered array" , opt.diagnostics)
          opt.jitter_psd_file = '%s/jexosim/data/Pointing/%s_pointing_model_psd.csv'%(opt.jexosim_path,opt.simulation.sim_pointing_psd.val)
          jexosim_msg ("psd file %s"%(opt.jitter_psd_file), opt.diagnostics)
          jexosim_msg ("running jitter code", opt.diagnostics)
          try:
              opt.use_external_jitter
          except AttributeError:
             opt.use_external_jitter=0
          if opt.use_external_jitter==0:
              jexosim_msg ("generating new jitter timeline...",  opt.diagnostics)
              opt.yaw_jitter, opt.pitch_jitter, opt.frame_osf = jexosim_lib.pointing_jitter(opt)
             
          elif opt.use_external_jitter==1:
              jexosim_msg ("using external jitter timeline...", opt.diagnostics)
              opt.yaw_jitter, opt.pitch_jitter, opt.frame_osf = opt.input_yaw_jitter, opt.input_pitch_jitter, opt._input_frame_osf
       
          jexosim_msg ("RMS jitter %s %s"%(np.std(opt.yaw_jitter), np.std(opt.pitch_jitter)  ) , opt.diagnostics)     
      
          pointing_timeline = create_pointing_timeline(opt) # this is also important to skip sections of jitter timeline that fall outside NDRs
              # the following takes into account skipped sections of jitter timeline due to reset groups
          opt.yaw_jitter_effective = pointing_timeline[:,1]*u.deg
          opt.pitch_jitter_effective = pointing_timeline[:,2]*u.deg              
          
          noise = simulate_jitter(opt)
            
  else:
     jexosim_msg ("using jitterless array", opt.diagnostics)
     jitterless = np.ones((fp.shape[0], fp.shape[1], len(opt.frames_per_ndr)))
     jitterless =  np.rollaxis(jitterless,2,0)
     jitterless = jitterless*fp
     jitterless =  np.rollaxis(jitterless,0,3)  
     jitterless = jitterless*opt.frames_per_ndr*opt.frame_time      
     noise = jitterless
     pointing_timeline= np.zeros((opt.ndr_end_frame_number[-1], 3))

  if opt.timeline.apply_lc.val ==0:
      jexosim_msg ("OMITTING LIGHT CURVE...", opt.diagnostics)
  else:
      jexosim_msg ("APPLYING LIGHT CURVE...", opt.diagnostics)
      noise *= opt.lc 
      
    #==============================================================================
    # add backgrounds
    #==============================================================================
    
  if (opt.background.EnableZodi.val == 1 or opt.background.EnableAll.val == 1) and opt.background.DisableAll.val != 1:
          jexosim_msg ("APPLYING ZODI..." , opt.diagnostics) 
          noise += opt.zodi.sed[opt.offs::opt.osf].reshape(-1,1) *\
        opt.frame_time * opt.frames_per_ndr   
  else:
      jexosim_msg ("NOT APPLYING ZODI..." , opt.diagnostics)

    
  if (opt.background.EnableEmission.val == 1 or opt.background.EnableAll.val == 1) and opt.background.DisableAll.val != 1:
          jexosim_msg ( "APPLYING EMISSION..."  ,  opt.diagnostics  )         
          noise += opt.emission.sed[opt.offs::opt.osf].reshape(-1,1) *\
            opt.frame_time * opt.frames_per_ndr        
  else:
      jexosim_msg ("NOT APPLYING EMISSION..." , opt.diagnostics)
           
#  plt.figure('zodi_test')
#  plt.plot(opt.zodi.sed[opt.offs::opt.osf], 'bx')
#
#  plt.figure('emission_test')
#  plt.plot(opt.emission.sed[opt.offs::opt.osf], 'bx', markersize = 10)

    #==============================================================================
    # add PRNU
    #==============================================================================
  if opt.noise.sim_prnu_rms.val ==0:
      opt.noise.ApplyPRNU.val =0
  
  if  (opt.noise.ApplyPRNU.val == 1 or opt.noise.EnableAll.val == 1) and opt.noise.DisableAll.val != 1:
  
      opt.qe_grid = opt.qe # used for flat field     
      jexosim_msg ("mean and standard deviation of flat field: should match applied flat in data reduction %s %s"%(opt.qe.mean(), opt.qe.std()),  opt.diagnostics  )
      jexosim_msg ("standard deviation of flat field uncertainty %s"%(opt.qe_uncert.std()), opt.diagnostics  )     
      applied_qe= opt.qe*opt.qe_uncert 
      
    #  qe_expand  = qe.repeat(3,axis=0)
    #  qe_expand  = qe_expand.repeat(3,axis=1)
    #  qe_expand /=9.0
    #  c_qe_expand = jexosim_lib.fast_convolution(qe_expand, 6e-06*pq.m, opt.kernel, opt.kernel_delta) 
    #  qe_cross_talk = c_qe_expand[1::3,1::3]
      
      jexosim_msg ("Applied QE (PRNU) grid std (includes uncertainty) %s"%(applied_qe.std()), opt.diagnostics)
      jexosim_msg ("APPLYING PRNU GRID...", opt.diagnostics  )
      noise =  np.rollaxis(noise,2,0)
      noise = noise*applied_qe
      noise =  np.rollaxis(noise,0,3)
         
  else:
      jexosim_msg ("PRNU GRID NOT APPLIED...", opt.diagnostics  )

    #==============================================================================
    # add dark signal
    #==============================================================================
                 
  if (opt.background.EnableDC.val ==1 or opt.background.EnableAll.val) and opt.background.DisableAll.val != 1:
      noise  += opt.channel.detector_pixel.Idc() * opt.frame_time * opt.frames_per_ndr 
          
      jexosim_msg ("DARK CURRENT being added...%s"%(opt.channel.detector_pixel.Idc.val ), opt.diagnostics  )
  else:
      jexosim_msg ("DARK CURRENT.. not... being added...", opt.diagnostics  )
           
  noise = np.where(noise >= 0.0, noise, 0)

    #==============================================================================
    # add shot noise
    #==============================================================================
 
  if (opt.noise.EnableShotNoise.val == 1 or opt.noise.EnableAll.val == 1) and opt.noise.DisableAll.val != 1:
        jexosim_msg ("PHOTON NOISE... being added...",  opt.diagnostics)
        aa =  time.time()
        cond =0 # keep this set to 0 until future possible parralelisation for large array sizes
        if cond ==0:       
            if opt.simulation.sim_full_ramps.val == 0 and opt.simulation.sim_use_UTR_noise_correction.val == 1:
                jexosim_msg ("applying correction to photon noise for UTR read", opt.diagnostics  )
                n = opt.projected_multiaccum
                noise =  poission_noise_UTR_correction(n, noise.value)
            else:
                noise = poission_noise(noise.value)
            noise = noise*u.electron  
            
        elif cond ==1: # for future possible parralelization
            pass
            # jexosim_msg('Splitting array for photon noise', opt.diagnostics)
            # z_size = int(1e7/(noise.shape[0]*noise.shape[1]))
            # n_div = int(noise.shape[2]/z_size)
            # for i in range(n_div):
            #     z0 = i*z_size
            #     if i == n_div-1:
            #         z1 = None
            #     else:
            #         z1 = z0+z_size
            #     div_noise = noise[...,z0:z1]
            #     if opt.simulation.sim_full_ramps.val == 0 and opt.simulation.sim_use_UTR_noise_correction.val == 1:
            #         jexosim_msg ("applying correction to photon noise for UTR read", opt.diagnostics  )
            #         n = opt.projected_multiaccum
            #         div_noise =  poission_noise_UTR_correction(n, div_noise.value)
            #     else:
            #         div_noise = poission_noise(div_noise.value)
                    
            #     if i ==0:
            #         div_noise_stack = div_noise
            #     else:
            #         div_noise_stack = np.dstack((div_noise_stack,div_noise))  
            # noise = div_noise_stack 
            # noise = noise*u.electron     
            
        jexosim_msg(f'Time to complete photon noise {time.time()-aa}', opt.diagnostics)
        
  else:  
       jexosim_msg ("PHOTON NOISE...NOT... being added...",  opt.diagnostics  )

  jexosim_msg ("check point 6.1 %s"%(noise.max()) , opt.diagnostics  )
 
    #==============================================================================
    # make ramps
    #==============================================================================

  for i in range(0, opt.n_ndr, opt.effective_multiaccum):
      noise[...,i:i+opt.effective_multiaccum] = np.cumsum(noise[...,i:i+opt.effective_multiaccum], axis=2)
 
    #==============================================================================
    # add read noise
    #==============================================================================

  if (opt.noise.EnableReadoutNoise.val == 1  or opt.noise.EnableAll.val == 1) and opt.noise.DisableAll.val != 1:  
        jexosim_msg ("READ NOISE... being added...", opt.diagnostics)
        if opt.simulation.sim_full_ramps.val == 0 and opt.simulation.sim_use_UTR_noise_correction.val == 1:             
            n = opt.projected_multiaccum
            jexosim_msg ("applying correction to read noise for UTR read", opt.diagnostics  )
            noise = read_noise_UTR_correction(n, noise.value , opt.channel.detector_pixel.sigma_ro.val)
            noise = noise*u.electron    
        else:    
            noise = read_noise(noise.value, opt.channel.detector_pixel.sigma_ro.val)
            noise = noise*u.electron    

  else:
       jexosim_msg ("READ NOISE...not... being added..." , opt.diagnostics  )
                
  jexosim_msg ("check point 7 %s %s"%(noise.max(), noise.min()) , opt.diagnostics  )

  return noise, pointing_timeline

#==============================================================================
# generates a signal only stack (containing noiseless star signal)
#==============================================================================
def signal_simulator(opt):
    
  fp_signal = opt.fp_signal[1::3,1::3]
  
  signal = np.ones((fp_signal.shape[0], fp_signal.shape[1], len(opt.frames_per_ndr)))
  signal =  np.rollaxis(signal,2,0)
  signal = signal*fp_signal
  signal =  np.rollaxis(signal,0,3)

  signal = signal*opt.frames_per_ndr*opt.frame_time
       
  if opt.timeline.apply_lc.val ==1:
      signal *= opt.lc 
  
  signal = np.where(signal >= 0.0, signal, 0)   
   
  jexosim_msg ("generating seperate noiseless signal array",  opt.diagnostics  )
  signal_ = np.zeros((signal.shape[0], signal.shape[1], signal.shape[2]))
  for i in range(0, signal.shape[2], opt.effective_multiaccum):
      for j in range(0, opt.effective_multiaccum):
          signal_[..., i+j] =  signal[..., i:i+j+1].sum(axis=2)  
  signal = signal_*u.electron 
 
  return signal

#==============================================================================
# run code
#==============================================================================

def run(opt):
 
  opt.fp =   opt.fp_original  # needed to reuse if cropped fp is used and monte carlo mode used
  opt.fp_signal =  opt.fp_signal_original # "
  
  opt.zodi.sed =  opt.zodi_sed_original # "
  opt.emission.sed =  opt.emission_sed_original # "

  opt.lc = opt.lc_original
  opt.ldc = opt.ldc_original
  opt.cr_wl = opt.cr_wl_original 
  opt.cr = opt.cr_original   
  opt.x_wav_osr = opt.x_wav_osr_original
  opt.x_pix_osr = opt.x_pix_osr_original
  opt.qe = opt.qe_original
  opt.qe_uncert = opt.qe_uncert_original
      
  opt.data , opt.pointing_timeline = noise_simulator(opt)
  if opt.pipeline.useSignal.val == 1:
      opt.data_signal_only = signal_simulator(opt) 
       
#==============================================================================
# this is very important! 
  if opt.noise.EnableSpatialJitter.val== 0:
        opt.pointing_timeline[:,2] =  opt.pointing_timeline[:,2]*0
  if opt.noise.EnableSpectralJitter.val== 0:
        opt.pointing_timeline[:,1] =  opt.pointing_timeline[:,1]*0   
  if opt.noise.DisableAll.val == 1:   
        opt.pointing_timeline[:,1] = opt.pointing_timeline[:,1]*0 
        opt.pointing_timeline[:,2] = opt.pointing_timeline[:,2]*0    
             
#========Apply gaps to Hi res NIRSpec arrays=================================== 
  if opt.channel.name == 'NIRSpec_BOTS_G140H_F100LP' or \
      opt.channel.name == 'NIRSpec_BOTS_G235H_F170LP'\
        or opt.channel.name == 'NIRSpec_BOTS_G395H_F290LP':   
     
      wav = opt.x_wav_osr[1::3]     
      idx0 = np.argwhere(wav==opt.wav_gap_start)[0].item() #recover the start of the gap
      idx1 = idx0 + opt.gap_len # gap of 172 pix works well for G1H and G2H, but a little off (by 0.01 microns for the other in some cases from published values)
      opt.data[:,idx0:idx1,:] = 0
      if opt.pipeline.useSignal.val == 1:
           opt.data_signal_only[:,idx0:idx1,:] = 0 
      if opt.diagnostics ==1:     
          import matplotlib.pyplot as plt
          plt.figure('show gap in focal plane image on an NDR')
          plt.imshow(opt.data[...,1].value,aspect='auto', interpolation = None)
          wl = opt.x_wav_osr.value[1::3]
          wl[idx0:idx1] = 0
          plt.figure('show gap in wl solution')
          plt.plot(wl, 'ro')
      
#==============================================================================    

  jexosim_plot('focal plane check1', opt.diagnostics, image=True, 
                  image_data=opt.fp_signal[1::3,1::3], aspect='auto', interpolation = None,
                  xlabel = 'x \'spectral\' pixel', ylabel = 'y \'spatial\' pixel')
  
  jexosim_plot('test - check NDR0', opt.diagnostics,
               image=True,  image_data = opt.data[...,0])
  jexosim_plot('test - check NDR1', opt.diagnostics,
               image=True,  image_data = opt.data[...,1])
  

  return opt
   
  
