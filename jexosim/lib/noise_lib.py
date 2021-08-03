"""
JexoSim 2.0

Noise library

"""

import numpy as np
from jexosim.lib.jexosim_lib import jexosim_msg, jexosim_plot
from jexosim.lib import jexosim_lib 
from astropy import units as u
import time
import scipy
from numba import jit, prange
import matplotlib.pyplot as plt

def apply_read_noise(signal, opt):   
    if opt.simulation.sim_full_ramps.val == 0 and opt.simulation.sim_use_UTR_noise_correction.val == 1:             
        n = opt.projected_multiaccum
        signal = read_noise_UTR_correction(n, signal.value , opt.channel.detector_pixel.sigma_ro.val)
        signal = signal*u.electron    
    else:    
        signal = read_noise(signal.value, opt.channel.detector_pixel.sigma_ro.val)
        signal = signal*u.electron
    return signal
      
def read_noise_UTR_correction (n, noise, sigma_ro):   
    # correction for utr noise: achieves same noise ar utr but using cds      
    sigma_ro_prime = np.sqrt(12.0*(n-1)/(n*(n+1)))*sigma_ro /np.sqrt(2)  
    noise = noise + np.random.normal(scale = sigma_ro_prime, size = noise.shape)
    return noise 

def read_noise (noise, sigma_ro):    
    noise = noise + np.random.normal(scale = sigma_ro, size = noise.shape)  
    return noise

def combine_noise(noise1, noise2):  # I don't think this is needed any more.
    sign = np.sign(noise1 + noise2)
    combined_noise = sign*(abs(noise1**2 + noise2**2))**0.5
    return combined_noise

def calc_fano_noise(quantum_yield, signal):   
    fano_factor = (3 * quantum_yield - quantum_yield**2 - 2) / quantum_yield   
    
    # old way
    # fano_noise = np.random.poisson(signal.value) - signal.value #photon noise   
    # fano_noise = fano_noise * np.sqrt(quantum_yield * fano_factor)  
    
    # new way - same result
    fano_noise = np.sqrt(signal)* np.sqrt(fano_factor*quantum_yield)
    fano_noise  = np.random.normal(0, fano_noise) * u.electron
    
    return fano_noise

def calc_poission_noise(signal):    
    noise = np.random.poisson(signal) - signal
    return noise 

def poission_noise_UTR_correction (n, noise):    
    alpha = 6.0*(n**2+1)/(5.0*n*(n+1))
    noise = noise*np.sqrt(alpha) #scale the noise
    return noise, alpha   

def apply_ipc (data, ipc_kernel):
    data = data.value
    for i in range(data.shape[2]):
        data[...,i] = scipy.signal.fftconvolve(data[...,i], ipc_kernel, 'same')
    data = data*u.electron
    return data  

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

def simulate_jitter(opt):       
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
    
    opt.pointing_timeline = create_pointing_timeline(opt) # this is also important to skip sections of jitter timeline that fall outside NDRs
    # the following takes into account skipped sections of jitter timeline due to reset groups
    opt.yaw_jitter_effective = opt.pointing_timeline[:,1]*u.deg
    opt.pitch_jitter_effective = opt.pointing_timeline[:,2]*u.deg                     
     
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
    else:
        mag = 1
              
    jexosim_msg ("mag %s"%(mag) , opt.diagnostics) 
    fp = jexosim_lib.oversample(fp, mag)
    osf *= mag
    offs = mag*offs + mag//2
    jitter_x *= mag
    jitter_y *= mag  
    jexosim_msg ("spatial osf %s"%(osf) , opt.diagnostics)    
    if opt.noise.EnableSpatialJitter.val ==0: 
              jitter_y *= 0.0
    if opt.noise.EnableSpectralJitter.val ==0:
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
    if cond == 1: 
        jexosim_msg('Splitting array for jitter code', 1)
        overlap = 5*osf
        x_size= int(5000/ (fp.shape[0]/osf) ) *osf
        n_div = int(fp.shape[1]/x_size)
        jexosim_msg(f'Number of divisions {n_div}', opt.diagnostics)
        ct = 0
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
              div_fp= fp[:, x0:x1]
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
              cc = time.time() -bb
              print(f'time remaining to complete jitter code {np.round((cc*(n_div-i)),2)} seconds', flush=True)
        jexosim_msg(f'noise stack shape after split jitter code {div_noise_stack.shape}', opt.diagnostics)  
        noise  = div_noise_stack                  
    qq = opt.frames_per_ndr* fp_units*opt.frame_time
    opt.signal = noise*qq 
    jexosim_msg('Time to run jitter code %s'%(time.time() - aa), opt.diagnostics) 
    return opt 

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
        
        import matplotlib.pyplot as plt
        aaa = np.arange(start, end, 1)
        plt.figure('pointing timeline - yaw')
        plt.plot(aaa,  opt.yaw_jitter[start:end])
        
    plt.plot(opt.yaw_jitter, 'g:')   
    
    return  pointingArray

def jitterless(opt):
   jexosim_msg ("using jitterless array", opt.diagnostics)
   blank_signal = np.ones((int(opt.fp.shape[0]/3), int(opt.fp.shape[1]/3), len(opt.frames_per_ndr)))
   jitterless =   (blank_signal.transpose() * opt.fp[1::3,1::3].transpose() ).transpose()   
   opt.signal = jitterless*opt.frames_per_ndr*opt.frame_time  
   opt.pointing_timeline= np.zeros((opt.ndr_end_frame_number[-1], 3))
   return opt 
   
def obtain_signal_only(opt):
   jexosim_msg ("generating seperate noiseless signal array",  opt.diagnostics  )
   fp = opt.fp_signal[1::3,1::3]
   blank_signal = np.ones((fp.shape[0], fp.shape[1], len(opt.frames_per_ndr)))
   signal =   (blank_signal.transpose() * fp.transpose() ).transpose() 
   signal = signal*opt.frames_per_ndr*opt.frame_time 

   if opt.timeline.apply_lc.val ==1:
      signal *= opt.lc
   signal = np.where(signal >= 0.0, signal, 0)  
   for i in range(0, opt.n_ndr, opt.effective_multiaccum):
      signal[...,i:i+opt.effective_multiaccum] = np.cumsum(signal[...,i:i+opt.effective_multiaccum], axis=2)
   if opt.simulation.sim_use_ipc.val == 1 and opt.channel.instrument.val !='MIRI':
       ipc_kernel = np.load(opt.channel.detector_array.ipc_kernel.val.replace('__path__', '%s/%s'%(opt.jexosim_path, 'jexosim')))
       signal =  apply_ipc(signal, ipc_kernel)
   signal = signal* u.electron
   return signal