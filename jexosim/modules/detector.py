"""
JexoSim 
2.0
Detector module
v1.0

"""

from jexosim.classes.sed import Sed
from jexosim.lib import jexosim_lib 
from jexosim.lib.jexosim_lib import jexosim_msg, jexosim_plot, planck
import numpy           as np
from astropy import units as u
import matplotlib.pyplot as plt
import copy, os
from scipy import interpolate
import sys
import scipy
 
def run(opt):
    
      opt.observation_feasibility = 1  # this variable is currently not changed to zero under any circumstances

#==============================================================================
#     get PSFs  
#==============================================================================

      if os.path.exists('%s/../archive/PSF/%s_psf_stack.npy'%(opt.__path__,opt.channel.instrument.val)):
          psf_stack = np.load('%s/../archive/PSF/%s_psf_stack.npy'%(opt.__path__,opt.channel.instrument.val))
          psf_stack_wl =  1e6*np.load('%s/../archive/PSF/%s_psf_stack_wl.npy'%(opt.__path__,opt.channel.instrument.val))
          psf = interpolate.interp1d(psf_stack_wl, psf_stack, axis=2,bounds_error=False, fill_value=0.0, kind='linear')(opt.x_wav_osr.value)      
          psf = np.rot90(psf) 
          opt.psf_type = 'wfe'
      else: # uses airy if no psf database however this is to be avoided, as pipeline assumes the wfe database psfs.
          psf = jexosim_lib.Psf(opt.x_wav_osr.value, opt.channel.camera.wfno_x.val, opt.channel.camera.wfno_y.val, opt.fp_delta.value, shape='airy')  
          psf[np.isnan(psf)] =0
          opt.psf_type = 'airy'

      opt.psf = psf 
      opt.channel.name
            
      # np.save('/Users/user1/Desktop/%s_psf_stack'%(opt.channel.name), psf[...,1::3])
      # np.save('/Users/user1/Desktop/%s_wl'%(opt.channel.name), opt.x_wav_osr[1::3].value)
      # xxxx

      jexosim_msg("PSF shape %s, %s"%(opt.psf.shape[0],opt.psf.shape[1] ), opt.diagnostics)     
      jexosim_plot('psf check', opt.diagnostics, image=True, image_data = psf[..., int(psf.shape[2]/2)])
    
      sum1=[]
      for i in range(psf.shape[2]):
          sum1.append(psf[...,i].sum())
          if psf[...,i].sum()!=0:
              # psf[...,i] =psf[...,i]/psf[...,i].sum() 
                if np.round(psf[...,i].sum(),3) !=1.0:    
                    jexosim_msg('error... check PSF normalisation %s %s'%(psf[...,i].sum(), opt.x_wav_osr[i]), 1)
                    sys.exit()
                  
      jexosim_plot('test7 - psf sum vs subpixel position (should be 1)', opt.diagnostics, 
                   xdata=opt.x_pix_osr, ydata = sum1, marker='bo')
    
#==============================================================================    
      # Populate focal plane with monochromatic PSFs
#============================================================================== 

      j0 = np.arange(opt.fp.shape[1]) - int(opt.psf.shape[1]/2)
      j1 = j0 + opt.psf.shape[1]
      idx = np.where((j0>=0) & (j1 < opt.fp.shape[1]))[0]
      i0 = np.array([opt.fp.shape[0]/2 - psf.shape[0]/2 + opt.offs]*len(j1)).astype(np.int)     
      i0+=1     
      
      # variable y position (applies only to NIRISS)
      if opt.y_pos_osr != []:
          opt.y_pos_osr = np.where(opt.y_pos_osr<0,0, opt.y_pos_osr).astype(np.int) +opt.fp.shape[0]/2 -opt.psf.shape[0]/2 + opt.offs 
          i0 = (opt.y_pos_osr).astype(np.int)
          
      i1 = i0 + psf.shape[0]
      
      # SPECIAL CASE: fix if airy psfs used     
      if opt.channel.name == 'NIRISS_SOSS_GR700XD':
          # fix if airy psfs used
          if opt.psf_type == 'airy':
              original_fp = copy.deepcopy(opt.fp)
              if i1.max() > opt.fp.shape[0]: #psfs will fall outside fp area due to curve
                  original_fp = copy.deepcopy(opt.fp)
                  opt.fp = np.zeros((i1.max(), opt.fp.shape[1] ))  # temp increase in fp size  
                  opt.fp_signal = np.zeros((i1.max(), opt.fp_signal.shape[1] ))  # temp increase in fp size           

      if opt.background.EnableSource.val == 1 or opt.background.EnableAll.val == 1:     
          for k in idx: # actual signal 
              opt.fp[i0[k]:i1[k], j0[k]:j1[k]] += psf[...,k] * opt.star.sed.sed[k].value 
              
      for k in idx: # used for getting sat time, and sizing
          opt.fp_signal[i0[k]:i1[k], j0[k]:j1[k]] += psf[...,k] * opt.star.sed.sed[k].value           
              
      # SPECIAL CASE: fix if airy psfs used      
      if opt.channel.name == 'NIRISS_SOSS_GR700XD': 
          if opt.psf_type == 'airy':  # now return fp to original size
              if i1.max() > original_fp.shape[0]: 
                  diff = i1.max()- original_fp.shape[0]         
                  opt.fp = opt.fp[diff:]    
                  opt.fp_signal = opt.fp_signal[diff:]   
                  
     
#==============================================================================
#     Now deal with the planet
#==============================================================================
      if opt.channel.name == 'NIRISS_SOSS_GR700XD':  # Niriss curved spectrum means code below will not work
          opt.planet.sed =  Sed(opt.x_wav_osr, opt.planet_sed_original)
      else:              
          i0p = np.unravel_index(np.argmax(opt.psf.sum(axis=2)), opt.psf[...,0].shape)[0]
          planet_response = np.zeros((opt.fp.shape[1]))         
          for k in idx: 
              planet_response[j0[k]:j1[k]] += psf[i0p,:,k] * opt.planet.sed.sed[k].value          
          pl_sed = np.zeros((opt.fp.shape[1]))
          for i in range (len(planet_response)): 
               pl_sed[i] = planet_response[i]/(1e-30+ opt.fp[:,i][(i0[i]+i1[i])//2])                   
          opt.planet.sed =  Sed(opt.x_wav_osr, pl_sed*u.dimensionless_unscaled)
                
      jexosim_plot('planet sed 1', opt.diagnostics, 
                   xdata=opt.planet.sed.wl, ydata = opt.planet.sed.sed, 
                   ylim=[0,1])

# alternate code that does not rely on multiplying by the star.sed first- not sure if this works correctly yet.    
#          planet_fp = copy.deepcopy(opt.fp)*0
#          i0 = np.array([opt.fp.shape[0]/2 - psf.shape[0]/2 + opt.offs]*len(j1))
#          i1 = i0 + psf.shape[0]
#          
#          for k in idx: 
#              planet_fp[i0[k]:i1[k], j0[k]:j1[k]] += psf[...,k] * opt.planet_sed_original[k] 
#
#  
#          planet_response = planet_fp.sum(axis=0)
#          plt.figure('planet sed 111')
#          plt.plot(opt.planet.sed.wl, planet_response)
#          
#          plt.figure('test')
#          plt.imshow(planet_fp)
          
#==============================================================================
#     Allocate pixel response function and convolve with focal plane
#==============================================================================
      kernel, kernel_delta = jexosim_lib.PixelResponseFunction(opt, 
        opt.psf.shape[0:2],
        7*opt.channel.simulation_factors.osf(),   
        opt.channel.detector_pixel.pixel_size(),
        lx = opt.channel.detector_pixel.pixel_diffusion_length())

      jexosim_msg ("kernel sum %s"%(kernel.sum()), opt.diagnostics) 
      jexosim_msg ("check 3.9 - unconvolved FP max %s"%(opt.fp.max()) , opt.diagnostics)      
      jexosim_plot('test3', opt.diagnostics, 
                   xdata=opt.x_wav_osr, ydata = opt.fp.sum(axis=0), marker='bo')
      jexosim_plot('test4', opt.diagnostics, 
                   xdata=opt.x_pix_osr, ydata = opt.x_wav_osr, marker='bo')                  
      jexosim_plot('test5', opt.diagnostics, 
                   xdata=opt.x_pix_osr, ydata = opt.fp.sum(axis=0), marker='bo')
      jexosim_plot('test6', opt.diagnostics, 
                   xdata=opt.x_wav_osr, ydata = opt.star.sed.sed, marker='bo') 
      
      opt.fp = jexosim_lib.fast_convolution(opt.fp, opt.fp_delta, kernel, kernel_delta)
      opt.fp_signal = jexosim_lib.fast_convolution(opt.fp_signal, opt.fp_delta, kernel, kernel_delta)

      jexosim_msg ("check 4 - convolved FP max %s"%(opt.fp.max()), opt.diagnostics)    #FPCOPY0 = exolib.fast_convolution(FPCOPY[1::3,1::3], 18e-6*pq.m, kernel, kernel_delta)    

      # jexosim_msg ("check 4 - convolved FP max %s %s"%(opt.fp.max(), FPCOPY[1::3,1::3].max()) , opt.diagnostics)    #FPCOPY0 = exolib.fast_convolution(FPCOPY[1::3,1::3], 18e-6*pq.m, kernel, kernel_delta)    
      opt.kernel = kernel
      opt.kernel_delta = kernel_delta
      
      # Fix units
      opt.fp = opt.fp*opt.star.sed.sed.unit 
      opt.fp_signal = opt.fp_signal*opt.star.sed.sed.unit  


#==============================================================================
#     Crop array to subarray
#============================================================================== 
      jexosim_msg ("now cropping to chosen subarray", opt.diagnostics)
      opt = user_subarray(opt)  
      
      if opt.fp_y == opt.fpn[0] and  opt.fp_x == opt.fpn[1]:    # do not crop as already right size     
            pass
      else: # set new subarray size for fp and fp_signal
         opt =  crop_to_subarray(opt)            


#==============================================================================
#     Find saturation time
#==============================================================================
    ## Find count rate with diffuse radiation
      FPCOPY = copy.deepcopy(opt.fp_signal)
      FPCOUNT_no_bkg = FPCOPY[1::3,1::3] * opt.quantum_yield.sed[1::3]
      FPCOPY += opt.zodi.sed  + opt.emission.sed 
      FPCOUNT = FPCOPY[1::3,1::3] * opt.quantum_yield.sed[1::3]
      FPCOUNT += opt.channel.detector_pixel.Idc.val   
      FPCOUNT = FPCOUNT.value
      FPCOUNT_no_bkg = FPCOUNT_no_bkg.value
      
      if opt.simulation.sim_use_ipc.val == 1 and opt.channel.instrument.val !='MIRI':
          ipc_kernel = np.load(opt.channel.detector_array.ipc_kernel.val.replace('__path__', '%s/%s'%(opt.jexosim_path, 'jexosim')))
          FPCOUNT = scipy.signal.fftconvolve(FPCOUNT, ipc_kernel, 'same') 
          FPCOUNT_no_bkg = scipy.signal.fftconvolve(FPCOUNT_no_bkg, ipc_kernel, 'same')
          # this will reduce the max count a bit.  ipc kernel not applied to images until noise module  
    
    ### apply gap for these modes to FPCOUNT and FPCOUNT_no_bkg for sat time measurement
      if opt.channel.name == 'NIRSpec_BOTS_G140H_F100LP' or \
        opt.channel.name == 'NIRSpec_BOTS_G235H_F170LP'\
            or opt.channel.name == 'NIRSpec_BOTS_G395H_F290LP':   
   
          idx0 = np.argwhere(opt.x_wav_osr[1::3] ==opt.wav_gap_start)[0].item() #recover the start of the gap
          idx1 = idx0 + opt.gap_len # gap of 172 pix works well for G1H and G2H, but a little off (by 0.01 microns for the other in some cases from published values)
          FPCOUNT[:, idx0:idx1] = 0
          FPCOUNT_no_bkg[:, idx0:idx1] = 0 
          # plt.figure('detector fp tex')
          # plt.imshow(FPCOUNT, aspect='auto')
          
          # plt.figure('detector fp 1d test')
          # plt.plot(opt.x_wav_osr[1::3], FPCOUNT.sum(axis=0) )
          
              
      jexosim_msg ("check 5 - %s"%(FPCOUNT.max()), opt.diagnostics)
      

      FW = opt.channel.detector_pixel.full_well.val 
      
      A,B = np.unravel_index(FPCOUNT.argmax(), FPCOUNT.shape)
      jexosim_msg ("maximum index and count with all backgrounds %s %s %s"%(A,B, FPCOUNT.max()), opt.diagnostics)
      
      A,B = np.unravel_index(FPCOUNT_no_bkg.argmax(), FPCOUNT_no_bkg.shape)
      jexosim_msg ("maximum index and count with no backgrounds %s %s %s"%(A,B, FPCOUNT_no_bkg.max()), opt.diagnostics)
      
      jexosim_msg ("full well in electron %s"%(FW), opt.diagnostics)   
      jexosim_msg ("saturation time assuming 100 percent full well with all backgrounds %s"%((FW / FPCOUNT.max())), opt.diagnostics)     
      jexosim_msg ("full well percentage chosen for saturation limit: %s"%(opt.observation.obs_fw_percent.val), opt.diagnostics)

      opt.sat_time = ((FW / FPCOUNT.max()) *opt.observation.obs_fw_percent.val/100.0 )*u.s
     
      opt.sat_time_no_bkg = ((FW / FPCOUNT_no_bkg.max()) *opt.observation.obs_fw_percent.val/100.0 )*u.s

      opt.sat_limit = u.electron*FW*opt.observation.obs_fw_percent.val/100.0  
      opt.sat_limit_fw = u.electron*FW
      
      jexosim_msg ('saturation limit %s'%(opt.sat_limit), opt.diagnostics)
        
      jexosim_msg ("saturation time with all backgrounds %s"%(opt.sat_time), opt.diagnostics)     
      jexosim_msg ("saturation time with no backgrounds or dc %s"%(opt.sat_time_no_bkg), opt.diagnostics)
     
      if opt.observation.obs_use_sat.val == 1: 
          jexosim_msg('Using saturation time to set n_groups', opt.diagnostics)
          n_groups = int(opt.sat_time/opt.t_g) # does not include reset group (assume this is after final read so saturation in this period does not affect read counts)
          if n_groups <2:
              n_groups=2
      else:
          jexosim_msg('Using user-defined n_groups', opt.diagnostics)
          n_groups = opt.observation.obs_n_groups.val
          
      jexosim_msg('Subarray used %s'%(opt.subarray), opt.diagnostics)
      jexosim_msg('Readout pattern used %s'%(opt.pattern), opt.diagnostics)
      jexosim_msg('t_f %s'%(opt.t_f), opt.diagnostics)
      jexosim_msg('t_g %s'%(opt.t_g), opt.diagnostics)
      jexosim_msg('dead time %s'%(opt.dead_time), opt.diagnostics)
      jexosim_msg('zero time %s'%(opt.zero_time), opt.diagnostics)
      jexosim_msg('n_groups %s'%(n_groups), opt.diagnostics)        
      
      # t_sim is not currently used, and by default is set to the same value as t_f
      opt.t_sim = opt.simulation.sim_t_sim.val*opt.t_f             
      opt.t_int =  (n_groups-1)*opt.t_g   
      opt.t_cycle = n_groups*opt.t_g+ opt.dead_time
        
      if n_groups*opt.t_g > opt.sat_time:
          jexosim_msg ("\n****** Warning!!!!!!!!!  : some pixels will exceed saturation limit ******\n", opt.diagnostics  )
          opt.sat_flag = 1
          # sys.exit()
           
      else:
          jexosim_msg ("\n****** OK!!!!!!  Cycle time within saturation time ******\n", opt.diagnostics  )
          opt.sat_flag = 0
  
#==============================================================================
# 10.  Set effective multiaccum
#==============================================================================
      if opt.simulation.sim_full_ramps.val == 0:
          jexosim_msg ("Approximating ramps with corrected CDS method, so only 2 NDRs simulated", 1)
          opt.effective_multiaccum = 2 # effective multiaccum is what is implemented in sim
          opt.projected_multiaccum = int(n_groups)
      else:
          opt.effective_multiaccum = int(n_groups)
          opt.projected_multiaccum = int(n_groups)
          
      jexosim_msg ("projected multiaccum: %s"%(opt.projected_multiaccum), opt.diagnostics)
      jexosim_msg ("effective multiaccum: %s"%(opt.effective_multiaccum), opt.diagnostics)
                                  
      opt.exposure_time = (opt.t_int + opt.dead_time + opt.zero_time) #same as t_cycle
      
      jexosim_msg ("Integration time - zeroth read %s"%(opt.t_int), opt.diagnostics)  
      jexosim_msg ("Estimated integration time incl. zeroth read %s"%(opt.t_int  + opt.zero_time), opt.diagnostics)
      jexosim_msg ("Estimated TOTAL CYCLE TIME %s"%(opt.exposure_time), opt.diagnostics)
           
      jexosim_msg ("CDS time estimate %s"%(opt.t_int), opt.diagnostics)  
        
      jexosim_msg ("FP max %s"%(opt.fp[1::3,1::3].max()), opt.diagnostics)
      jexosim_msg ("DC SWITCH.......%s"%(opt.background.EnableDC.val), opt.diagnostics)
      jexosim_msg ("DISABLE ALL SWITCH.......%s"%(opt.background.DisableAll.val), opt.diagnostics)   
      jexosim_msg ("DARK CURRENT %s"%(opt.channel.detector_pixel.Idc.val) , opt.diagnostics)
  
      jexosim_plot('focal plane check', opt.diagnostics, image=True, 
                  image_data=opt.fp[1::3,1::3], aspect='auto', interpolation = None,
                  xlabel = 'x \'spectral\' pixel', ylabel = 'y \'spatial\' pixel')
      if opt.diagnostics ==1:
          plt.figure('focal plane check')
          cbar = plt.colorbar()
          cbar.set_label(('Count (e$^-$/s)'), rotation=270, size=15,labelpad=20)
          cbar.ax.tick_params(labelsize=15) 
          ax = plt.gca()
          for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
              item.set_fontsize(15)

      opt.fp_original = copy.deepcopy(opt.fp)
      opt.fp_signal_original = copy.deepcopy(opt.fp_signal)  
      opt.x_wav_osr_original = copy.deepcopy(opt.x_wav_osr)
      opt.x_pix_osr_original = copy.deepcopy(opt.x_pix_osr) 
      opt.quantum_yield_original = copy.deepcopy(opt.quantum_yield)
      
      opt.zodi_sed_original = copy.deepcopy(opt.zodi.sed) # needed here due to possible cropping above for subarrays
      opt.emission_sed_original = copy.deepcopy(opt.emission.sed)
       
      if opt.channel.instrument.val =='NIRSpec':
         opt.channel.pipeline_params.wavrange_hi.val = opt.gap[3]
         opt.channel.pipeline_params.wavrange_lo.val = opt.gap[2]
         opt.channel.pipeline_params.end_wav.val = opt.gap[3]+0.1
         opt.channel.pipeline_params.start_wav.val = opt.gap[2]-0.1
  
      jexosim_plot('final wl solution on subarray', opt.diagnostics,
                   ydata=opt.x_wav_osr[1::3],
                   xlabel = 'x \'spectral\' pixel', ylabel = 'y \'spatial\' pixel',
                   grid=True)
     
      if opt.diagnostics ==1: 
          
          wl = opt.x_wav_osr  
          wav = opt.x_wav_osr[1::3]   
          
          if opt.channel.name == 'NIRSpec_BOTS_G140H_F100LP' or \
              opt.channel.name == 'NIRSpec_BOTS_G235H_F170LP'\
                  or opt.channel.name == 'NIRSpec_BOTS_G395H_F290LP':   
         
              idx0 = np.argwhere(wav==opt.wav_gap_start)[0].item() #recover the start of the gap
              idx1 = idx0 + opt.gap_len # gap of 172 pix works well for G1H and G2H, but a little off (by 0.01 microns for the other in some cases from published values)
              fp_1d = opt.fp_signal[1::3,1::3].sum(axis=0)
              fp_1d[idx0:idx1] = 0

              idx0 = np.argwhere(wl==opt.wav_gap_start)[0].item() #recover the start of the gap
              idx1 = idx0 + opt.gap_len*3 # gap of 172 p        
              opt.PCE[idx0:idx1] = 0
              opt.TotTrans[idx0:idx1] = 0
              opt.R.sed[idx0:idx1] = 0
             
          else:
              fp_1d = opt.fp_signal[1::3,1::3].sum(axis=0)
              
          plt.figure('Focal plane pixel count rate')
          plt.plot(wav, fp_1d, 'r-')
          plt.grid()

          plt.figure('Final PCE and transmission on subarray')          
          plt.plot(wl, opt.PCE, '-') 
          plt.plot(wl, opt.TotTrans, '--') 
          
          plt.figure('R power')
          plt.plot(wl, opt.R.sed, '-') 

          # xxxx
           # np.save('/Users/user1/Desktop/PCE_%s_%s.npy'%(opt.channel.name, opt.subarray), opt.PCE.value)
           # np.save('/Users/user1/Desktop/TotTrans_%s_%s.npy'%(opt.channel.name, opt.subarray), opt.TotTrans.value)
           # np.save('/Users/user1/Desktop/xwavosr_%s_%s.npy'%(opt.channel.name, opt.subarray), opt.x_wav_osr.value)
           # np.save('/Users/user1/Desktop/R_%s_%s.npy'%(opt.channel.name, opt.subarray), opt.R.sed.value)

          # 
      sanity_check(opt)
      
   
      return opt
# =============================================================================
#       
# =============================================================================
      
def sanity_check(opt):
    import scipy.constants as spc
    
    wl = opt.x_wav_osr[1::3]
    del_wl = abs(np.gradient(wl))
#    del_wl = opt.d_x_wav_osr[1::3]*3
    star_spec = opt.star_sed
    star_spec.rebin(wl)
    T = opt.planet.planet.star.T
    trans_sed = opt.total_transmission.sed*u.dimensionless_unscaled
    trans = Sed(opt.total_transmission.wl,trans_sed)
    trans.rebin(wl)
    QE = opt.qe_spec
    QE.rebin(wl)
    quantum_yield = opt.quantum_yield
    quantum_yield.rebin(wl)
    Rs = (opt.planet.planet.star.R).to(u.m)
    D = (opt.planet.planet.star.d).to(u.m)
    n= quantum_yield.sed* trans.sed*del_wl*np.pi*planck(wl,T)*(Rs/D)**2*opt.Aeff*QE.sed/(spc.h*spc.c/(wl*1e-6))

    n2= quantum_yield.sed* trans.sed*del_wl*star_spec.sed*opt.Aeff*QE.sed/(spc.h*spc.c/(wl*1e-6))
    
    jex_sig = quantum_yield.sed*opt.fp_signal[1::3,1::3].sum(axis=0)
    R = opt.pipeline.pipeline_R.val
    del_wav = wl/R
    opt.exp_sig  = opt.t_int*del_wav*jex_sig/del_wl
    
    if opt.diagnostics ==1:
        plt.figure('sanity check 1 - check focal plane signal')
        plt.plot(wl,n, 'b^', label='BB check')
        plt.plot(wl,n2, 'r+', label='Phoenix check')  # not convolved with PSF unlike JexoSim, so peak may be higher
        plt.plot(wl, jex_sig, 'gx', label='JexoSim')
        plt.ylabel('e/s/pixel col'); plt.xlabel('pixel col wavelength (microns)')
        plt.legend(loc='best')
        
        plt.figure('sanity check 2 - expected final star signal in R bin of %s'%((R)))
        plt.plot(wl, opt.exp_sig)
        plt.ylabel('e/bin'); plt.xlabel('Wavelength (microns)')
            
        plt.figure('sanity check 3 - expected photon noise (sd) in R bin of %s'%((R)))
        plt.plot(wl, opt.exp_sig**0.5)    
        plt.ylabel('e/bin'); plt.xlabel('Wavelength (microns)')  
       

def user_subarray(opt):
    
    s = (opt.observation.obs_inst_config.val).replace(" ", "")
    start = 0
    idx=[]
    for i in range(3):
         idx0 = s.find('+', start)
         idx.append(idx0)
         start =idx0+1
    opt.subarray = s[idx[1]+1:idx[2]]
    opt.pattern = s[idx[2]+1:]

    idx1 = np.argwhere(np.array(opt.channel.detector_array.subarray_list.val) == opt.subarray)[0].item()
    idx2 = np.argwhere(np.array(opt.channel.detector_readout.pattern_list.val) == opt.pattern)[0].item()
    opt.nframes = opt.channel.detector_readout.pattern_params_list.val[idx2][0]
    opt.nskip = opt.channel.detector_readout.pattern_params_list.val[idx2][1]
    opt.fp_x = opt.channel.detector_array.subarray_geometry_list.val[idx1][1]
    opt.fp_y = opt.channel.detector_array.subarray_geometry_list.val[idx1][0]
    opt.t_f = opt.channel.detector_array.subarray_t_f_list.val[idx1]

    opt.t_g = (opt.nframes+opt.nskip)*opt.t_f 
    opt.dead_time = (opt.channel.detector_readout.nGND.val+ opt.channel.detector_readout.nRST.val)* opt.t_g
    opt.zero_time = opt.channel.detector_readout.nNDR0.val* opt.t_g
    
    if opt.channel.instrument.val =='NIRSpec': 
            opt.gap = opt.channel.detector_array.subarray_gap_list.val[idx1] 

    return opt
 
   
def crop_to_subarray(opt):

    opt.fpn[0] = opt.fp_y*u.dimensionless_unscaled
    opt.fpn[1] = opt.fp_x*u.dimensionless_unscaled
            
    ycrop = int(((opt.fp.shape[0]-opt.fpn[0]*3)/2).value)
    ycrop0 = -int(((opt.fp.shape[0]-opt.fpn[0]*3)/2).value)
                              
    xcrop = int(((opt.fp.shape[1]-opt.fpn[1]*3)/2).value)
    xcrop0 = -int(((opt.fp.shape[1]-opt.fpn[1]*3)/2).value)
    
    if opt.fpn[0] == opt.fp.shape[0]/3:
        ycrop=0; ycrop0=None
  
    if opt.fpn[1] == opt.fp.shape[1]/3:
        xcrop=0; xcrop0=None 
 
    cond=0    
    # special case for NIRSpec subarrays
    

    if opt.channel.instrument.val =='NIRSpec':
        #wavelength solutions centered at division of subarrays to give right wavelength in each subarray    
        if opt.channel.name != 'NIRSpec_BOTS_G140H_F100LP' \
            and opt.channel.name != 'NIRSpec_BOTS_G395H_F290LP'  and opt.channel.name != 'NIRSpec_BOTS_G235H_F170LP' :
            #just use one detector
            if opt.subarray == 'SUB1024B':
                xcrop =1024*3
                xcrop0=None
            if opt.subarray == 'SUB1024A':
                xcrop =0
                xcrop0 =-1024*3
            if opt.subarray == 'SUB512' or opt.subarray == 'SUB512S':
                xcrop =1024*3
                xcrop0 =-512*3
        else:  # H modes start off with large array 4500 in x which we crop down      
            # idx = np.argwhere((opt.x_wav_osr[1::3].value>=opt.gap[0]) & (opt.x_wav_osr[1::3].value<=opt.gap[1]))
            # jexosim_msg('gap width in whole pixels: %s'%(len(idx)), opt.diagnostics)
      
            # if opt.subarray == 'SUB2048':
            #     xcrop= 0
            #     xcrop0 = 4096*3+len(idx)*3
                
            # if opt.subarray == 'SUB1024A':
            #     xcrop=0
            #     xcrop0= 2048*3+len(idx)*3
            
            # if opt.subarray == 'SUB1024B':    
            #     xcrop= 1024*3
            #     xcrop0 = 1024*3+ 2048*3+len(idx)*3
            # if opt.subarray == 'SUB512' or opt.subarray == 'SUB512S':    
            #     xcrop= 1024*3
            #     xcrop0 = 1024*3+ 512*3  +len(idx)*3 + 512*3
                
            # idx = np.argwhere((opt.x_wav_osr[1::3].value>=opt.gap[0]) & (opt.x_wav_osr[1::3].value<=opt.gap[1]))
            # jexosim_msg('gap width in whole pixels: %s'%(len(idx)), opt.diagnostics)
            gap = 172 # tried 157 based on 2048 subarrays and plotting the published gaps using above line, but with G2H a little short on NRS2; 172 seems to work best to obtain the gap values across all grisms that match the published values
      
            if opt.subarray == 'SUB2048':
                xcrop= 0
                xcrop0 = 4096*3+gap*3
                gap_idx0 = 2048
                opt.gap_len = gap
                
            if opt.subarray == 'SUB1024A':
                xcrop=0
                xcrop0= 2048*3+ 2048*3 + gap*3
                gap_idx0 = 1024
                opt.gap_len = 1024*2 +gap               
            
            if opt.subarray == 'SUB1024B':    
                xcrop= 1024*3
                xcrop0 = xcrop + 2048*3+ gap*3
                gap_idx0 = 1024
                opt.gap_len = gap          
                
            if opt.subarray == 'SUB512' or opt.subarray == 'SUB512S':    
                xcrop= 1024*3
                xcrop0 = xcrop + 2*512*3  +gap*3 + 2*512*3
                gap_idx0 = 512
                opt.gap_len = 512*2 + gap #length of gap in pix
                
            wav = opt.x_wav_osr[xcrop:xcrop0]
            wav = wav[1::3]
            opt.wav_gap_start = wav[gap_idx0] # first wavelength in gap - use to identify start of gap in noise after cropping
            opt.wav_gap_end = wav[gap_idx0+opt.gap_len] # first wavelength in gap - use to identify start of gap in noise after cropping
 
            cond=1
    
    # special case for NIRISS substrip 96
    if opt.subarray =='SUBSTRIP96' and opt.channel.instrument.val =='NIRISS':
          if opt.psf_type != 'airy': # should not really use airy however - potential problem in data reduction as designed for wfe psf.
              ycrop = 360; ycrop0= -120 
              xcrop =0
              xcrop0 = None
              
    # if ycrop > 0 or ycrop0 >0: 
    opt.fp = opt.fp[ycrop:ycrop0]
    opt.fp = opt.fp[:,xcrop:xcrop0]
    
    opt.fp_signal = opt.fp_signal[ycrop:ycrop0]
    opt.fp_signal = opt.fp_signal[:,xcrop:xcrop0]  
    
    opt.x_wav_osr = opt.x_wav_osr[xcrop:xcrop0]
    opt.d_x_wav_osr = opt.d_x_wav_osr[xcrop:xcrop0]
    opt.x_pix_osr = opt.x_pix_osr[xcrop:xcrop0]
    opt.zodi.sed = opt.zodi.sed[xcrop:xcrop0]
    opt.emission.sed = opt.emission.sed[xcrop:xcrop0]
    opt.zodi.wl = opt.x_wav_osr
    opt.emission.wl = opt.x_wav_osr
    opt.planet.sed.sed =opt.planet.sed.sed[xcrop:xcrop0]                  
    opt.planet.sed.wl = opt.x_wav_osr
    opt.R.sed = opt.R.sed[xcrop:xcrop0] 
    opt.R.wl = opt.x_wav_osr
    opt.quantum_yield.sed = opt.quantum_yield.sed[xcrop:xcrop0] 
    opt.quantum_yield.wl = opt.x_wav_osr

    
    opt.PCE = opt.PCE[xcrop:xcrop0]     
    opt.TotTrans = opt.TotTrans[xcrop:xcrop0]  
    
    if cond==1: #fix for resized NIRSpec Hi-res arrays with gaps
        opt.fpn[0] = opt.fp.shape[0]/3 
        opt.fpn[1] = opt.fp.shape[1]/3  

    if opt.fp.shape[0]/3 != opt.fpn[0]:
        jexosim_msg('Error: detector 1 - check code', 1)
        sys.exit()
    if opt.fp.shape[1]/3 != opt.fpn[1]: 
        jexosim_msg('Error: detector 2 - check code', 1)
        sys.exit()
    jexosim_msg ("subarray dimensions %s x %s "%(opt.fpn[0], opt.fpn[1]), 1)
    
    return opt
        
