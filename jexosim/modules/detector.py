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
 

def run(opt):
    
      opt.observation_feasibility = 1  # if saturates before cycle time then this is changed to 0

#==============================================================================
# #    get PSFs  
#==============================================================================
      ch = opt.channel

      if os.path.exists('%s/../databases/PSF/%s_psf_stack.npy'%(opt.__path__,ch.instrument.val)):    
          psf_stack = np.load('%s/../databases/PSF/%s_psf_stack.npy'%(opt.__path__,ch.instrument.val))   
          psf_stack_wl =  1e6*np.load('%s/../databases/PSF/%s_psf_stack_wl.npy'%(opt.__path__,ch.instrument.val))       
          psf = interpolate.interp1d(psf_stack_wl, psf_stack, axis=2,bounds_error=False, fill_value=0.0, kind='linear')(opt.x_wav_osr.value)      
          psf = np.rot90(psf)       
      else:              
          psf = jexosim_lib.Psf(opt.x_wav_osr, ch.camera.wfno_x(), ch.camera.wfno_y(), opt.fp_delta, shape='airy')  
      opt.psf = psf 

      jexosim_msg("PSF shape %s, %s"%(opt.psf.shape[0],opt.psf.shape[1] ), opt.diagnostics)     
      jexosim_plot('psf check', opt.diagnostics, image=True, image_data = psf[..., int(psf.shape[2]/2)])
    
      sum1=[]
      for i in range(psf.shape[2]):
          sum1.append(psf[...,i].sum())
          if psf[...,i].sum()!=0:
#              psf[...,i] =psf[...,i]/psf[...,i].sum() 
              if np.round(psf[...,i].sum(),3) !=1.0:    
                  jexosim_msg('error... check PSF normalisation %s'%(psf[...,i].sum()), 1)
                  sys.exit()
                  
      jexosim_plot('test7 - psf sum vs subpixel position (should be 1)', opt.diagnostics, 
                   xdata=opt.x_pix_osr, ydata = sum1, marker='bo')
          
#==============================================================================    
           #7# Populate focal plane with monochromatic PSFs
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
    
      FPCOPY = copy.deepcopy(opt.fp)
      opt.fp_signal = copy.deepcopy(opt.fp)   
      for k in idx:
          FPCOPY[i0[k]:i1[k], j0[k]:j1[k]] += opt.psf[...,k] * opt.star.sed.sed[k].value    
      if opt.background.EnableSource.val == 1 or opt.background.EnableAll.val == 1:
          for k in idx: 
              opt.fp[i0[k]:i1[k], j0[k]:j1[k]] += psf[...,k] * opt.star.sed.sed[k].value          
      for k in idx: 
              opt.fp_signal[i0[k]:i1[k], j0[k]:j1[k]] += psf[...,k] * opt.star.sed.sed[k].value                
      opt.pre_convolved_fp = copy.deepcopy(opt.fp)
#==============================================================================
# #8# Now deal with the planet
#==============================================================================
      if opt.channel.name == 'NIRISS_SOSS_ORDER_1':  # Niriss curved spectrum means code below will not work
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
# 9# Allocate pixel response function and conolve with focal plane
#==============================================================================

      kernel, kernel_delta = jexosim_lib.PixelResponseFunction(opt, 
        opt.psf.shape[0:2],
        7*ch.simulation_factors.osf(),   
        ch.detector_pixel.pixel_size(),
        lx = ch.detector_pixel.pixel_diffusion_length())

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
      FPCOPY = jexosim_lib.fast_convolution(FPCOPY, opt.fp_delta, kernel, kernel_delta)       
      opt.fp_signal = jexosim_lib.fast_convolution(opt.fp_signal, opt.fp_delta, kernel, kernel_delta)
      
      jexosim_msg ("check 4 - convolved FP max %s %s"%(opt.fp.max(), FPCOPY[1::3,1::3].max()) , opt.diagnostics)    #FPCOPY0 = exolib.fast_convolution(FPCOPY[1::3,1::3], 18e-6*pq.m, kernel, kernel_delta)    

      opt.kernel = kernel
      opt.kernel_delta = kernel_delta
      
      # Fix units
      opt.fp = opt.fp*opt.star.sed.sed.unit  
      opt.fp_signal = opt.fp_signal*opt.star.sed.sed.unit  

#==============================================================================
# 10.  Find saturation time
#==============================================================================
    ## Find count rate with diffuse radiation 
      FPCOPY += opt.zodi.sed.value   + opt.emission.sed.value  
      FPCOUNT = FPCOPY[1::3,1::3] + ch.detector_pixel.Idc.val.value    
      FPCOUNT = FPCOUNT*u.electron/u.s 
    
      jexosim_msg ("check 5 - %s"%(FPCOUNT.max()), opt.diagnostics)

      FW = ch.detector_pixel.full_well.val*u.electron   
      A,B = np.unravel_index(FPCOUNT.argmax(), FPCOUNT.shape)
      jexosim_msg ("maximum index and count with all backgrounds %s %s %s"%(A,B, FPCOUNT.max()), opt.diagnostics)
      A,B = np.unravel_index(opt.fp_signal[1::3,1::3].argmax(), opt.fp_signal[1::3,1::3].shape)
      jexosim_msg ("maximum index and count with no backgrounds %s %s %s"%(A,B, opt.fp_signal[1::3,1::3].max()), opt.diagnostics)
      jexosim_msg ("full well %s"%(FW), opt.diagnostics)   
      jexosim_msg ("full cycle time based on 100 percent saturation %s"%((FW / FPCOUNT.max())), opt.diagnostics)     
      jexosim_msg ("maximum full well percentage set to: %s"%(opt.channel.detector_readout.pc_fw.val), opt.diagnostics)
             
      sat_time = ((FW / FPCOUNT.max()).value *opt.channel.detector_readout.pc_fw.val/100.0 )*u.s
      opt.sat_time = sat_time
      
      jexosim_msg ("saturation time adjusted for maximum full well percentage %s"%(sat_time), opt.diagnostics)     
      jexosim_msg ("saturation time with no backgrounds or dc %s"%( (FW/(opt.fp_signal[1::3,1::3].max()))*opt.channel.detector_readout.pc_fw.val/100.0), opt.diagnostics)

#==============================================================================
# 10.  Fix t_f and n (frame time and multaccum) - assume t_g = t_f for non-Fowler sampling m=1
#==============================================================================
      
      #dead time is the duration for reset and idle
      dead_time = (ch.detector_readout.nGND.val+ ch.detector_readout.nRST.val)* ch.detector_readout.t_g.val
      #zero time is duration for first group (first frame in non-MCDS), subtracted from total integration time
      zero_time = ch.detector_readout.nNDR0.val* ch.detector_readout.t_g.val

    #==============================================================================
    #       If using saturation time
    #==============================================================================
      if ch.detector_readout.use_sat.val == 1:  #uses the calculated multiaccum to set t_int
          jexosim_msg ("saturation time implemented", opt.diagnostics)   
          if len(opt.channel.detector_array.subarray_list.val) > 1: 
              cond=1
          else:
              cond=0
            #==============================================================================
            #       If multiple subarray modes exist, pick the one with best SNR
            #==============================================================================        
        
          if cond ==1: # different subarrays exist
              jexosim_msg ("Selecting optimal subarray...", 1) #assuming m=1 in all cases 
              SNR_list=[]
              for i in range(len(opt.channel.detector_array.subarray_list.val)):
                                               
                      t_f_subarray = opt.channel.detector_array.subarray_t_f_list.val[i]
                      
                      dead_time = (ch.detector_readout.nGND.val+ ch.detector_readout.nRST.val)* t_f_subarray                      
                      t_crit = 2*t_f_subarray + dead_time # min cycle time  
                      
                      # assume that sat time must include the dead time
                      n_groups =  int(sat_time/t_f_subarray) - int(dead_time/t_f_subarray)
                      #"An exposure consists of consecutive integrations, separated by a reset of the detector where the pixels are reset individually at the same cadence at which they are read out." 
                      jexosim_msg ("dead time frames %s"%(dead_time/t_f_subarray), opt.diagnostics)
                      
                      if n_groups < 2: # must have at least 2 groups
                          SNR=0
                      elif t_crit > sat_time:  #  saturates within min cycle time
                          SNR = 0                                                    
                      else:
                          t_int =  (n_groups-1)*t_f_subarray   
                          f = FPCOUNT.max()  # max count rate on detector (includes dc and backgrounds)     
                          # see Rauschser and Fox 2007
                          RN_var = 12.*(n_groups-1)*(opt.channel.detector_pixel.sigma_ro.val)**2 /(n_groups*(n_groups+1))
                          PN_var = ((6.*(n_groups**2+1)/(5*n_groups*(n_groups+1))) * t_int * f).value
                          TN= np.sqrt(RN_var+PN_var) #total noise
                          S = (t_int * f).value  # total signal (assumes first frame count is subtracted)
                          SNR = S/TN
                   
                      if SNR ==0: RN_var=PN_var=S=0 
                      jexosim_msg ( "%s t_f %s t_cycle_min %s "%(opt.channel.detector_array.subarray_list.val[i],   t_f_subarray,  t_crit) ,opt.diagnostics)
                      jexosim_msg ("n: %s RN %s PN %s signal %s SNR %s"%(n_groups,   RN_var**0.5,  PN_var**0.5,   S,  SNR ) ,opt.diagnostics)
                      SNR_list.append(SNR)              
               
              idx = np.argmax(np.array(SNR_list))
              new_t_f =  opt.channel.detector_array.subarray_t_f_list.val[idx]
              jexosim_msg ("Using subarray mode %s"%(opt.channel.detector_array.subarray_list.val[idx]), 1)
              opt.subarray = opt.channel.detector_array.subarray_list.val[idx]
              
              if np.array(SNR_list).max() ==0:
                  jexosim_msg ("!!! No subarray modes will work for this object", 1)
                  opt.observation_feasibility = 0
                                 
              opt.channel.detector_readout.t_f.val = new_t_f
              opt.channel.detector_readout.t_g.val = new_t_f
              opt.channel.detector_readout.t_sim.val = new_t_f
              jexosim_msg ("frame time (t_f) is now... %s"% (new_t_f), 1)
              dead_time = (ch.detector_readout.nGND.val+ ch.detector_readout.nRST.val)* ch.detector_readout.t_g.val
              zero_time = ch.detector_readout.nNDR0.val* ch.detector_readout.t_g.val                  
          
              fpn = opt.channel.detector_array.subarray_geometry_list.val[idx]

              if fpn[0] == opt.fpn[0] and  fpn[1] == opt.fpn[1]:         
                  pass
              else: # set new subaary size for FPA
                  opt.fpn[0] = np.array(fpn[0])*u.dimensionless_unscaled
                  opt.fpn[1] = np.array(fpn[1])*u.dimensionless_unscaled
                  
                  ycrop = int(((opt.fp_signal.shape[0]-opt.fpn[0]*3)/2).value)
                  ycrop0 = int(((opt.fp_signal.shape[0]-opt.fpn[0]*3)/2).value)
                                    
                  xcrop = int(((opt.fp_signal.shape[1]-opt.fpn[1]*3)/2).value)
                  xcrop0 = int(((opt.fp_signal.shape[1]-opt.fpn[1]*3)/2).value)
                  
                  # special case for NIRISS substrip 96
                  if opt.channel.detector_array.subarray_list.val[idx] =='SUBSTRIP96' and \
                              opt.channel.instrument.val =='NIRISS':
                        ycrop = 360; ycrop0= 120 
                        xcrop0 = xcrop =0
                
                  if ycrop > 0 or ycrop0 >0: 
                      opt.fp_signal = opt.fp_signal[ycrop:-ycrop0] # crop the oversampled FP array to 96 x 3 in y axis
                      opt.fp = opt.fp[ycrop:-ycrop0]
         
                  if xcrop > 0 or xcrop0 >0:     
                      opt.fp_signal = opt.fp_signal[:,xcrop:-xcrop0] # crop the oversampled FP array to 96 x 3 in y axis
                      opt.fp = opt.fp[:,xcrop:-xcrop0]
                      opt.x_wav_osr = opt.x_wav_osr[xcrop:-xcrop0]
                      opt.d_x_wav_osr = opt.d_x_wav_osr[xcrop:-xcrop0]
                      opt.x_pix_osr = opt.x_pix_osr[xcrop:-xcrop0]
                      opt.zodi.sed = opt.zodi.sed[xcrop:-xcrop0]
                      opt.emission.sed = opt.emission.sed[xcrop:-xcrop0]
                      opt.zodi.wl = opt.x_wav_osr
                      opt.emission.wl = opt.x_wav_osr
                      opt.planet.sed.sed =opt.planet.sed.sed[xcrop:-xcrop0]                  
                      opt.planet.sed.wl = opt.x_wav_osr
                      
                  if opt.fp_signal.shape[0]/3 != opt.fpn[0]:
                      xxx
                  if opt.fp_signal.shape[1]/3 != opt.fpn[1]: 
                      xxx
                  jexosim_msg ("subarray dimensions %s x %s "%(opt.fpn[0], opt.fpn[1]), 1)
                  
            #==============================================================================
            #       If multiple subarray modes do not eixst, t_f, t_g, dead_time and zero_time not changed 
            #==============================================================================              
          elif cond ==0: # different subbarrays do not exist,\        
              jexosim_msg ("Only one subarray available...",  1) #assuming m=1 in all cases 
              opt.subarray = opt.channel.detector_array.subarray_list.val[0]
              jexosim_msg ("Using subarray %s"%(opt.channel.detector_array.subarray_list.val[0]) , 1)
                  
          # re-calculate or calculate n (maccum_calc) and t_int
          # print (sat_time, opt.channel.detector_readout.t_g.val, dead_time)
          jexosim_msg ("MACCUM (no of groups) %s"%(int(sat_time/opt.channel.detector_readout.t_g.val) - int(dead_time/ch.detector_readout.t_g.val)) , opt.diagnostics  )
          maccum_calc =  int(sat_time/opt.channel.detector_readout.t_g.val) - int(dead_time/ch.detector_readout.t_g.val)
          t_int =  (maccum_calc-1)*ch.detector_readout.t_g.val        
     
    #==============================================================================
    #       If NOT using saturation time, t_f, t_g, dead_time and zero_time not changed 
    #==============================================================================
      
      elif ch.detector_readout.use_sat.val == 0:  #uses the chosen multiaccum to set t_int, t_g is fixed by the ICF
          jexosim_msg ("not using saturation time...", opt.diagnostics)
          #  load n (maccum_calc) and calculate t_int
          maccum_calc = int(ch.detector_readout.multiaccum.val)
          t_int = (ch.detector_readout.t_g.val*(ch.detector_readout.multiaccum.val-1))  # int time = cds times = ndr1 time
      
      jexosim_msg ("t_int %s"%(t_int), opt.diagnostics)
      jexosim_msg ("frame time %s t_dead %s t_zero %s"%(opt.channel.detector_readout.t_f.val,  dead_time,   zero_time) , opt.diagnostics)
      
      opt.t_int = t_int
      jexosim_msg ("multiaccum projected %s"%(maccum_calc), 1)
            
      if ch.detector_readout.doCDS.val == 1:
          jexosim_msg ("Using CDS, so only 2 NDRs simulated", opt.diagnostics)
          opt.effective_multiaccum = 2 # effective multiaccum is what is implemented in sim
          opt.projected_multiaccum = maccum_calc 
      else:
          opt.effective_multiaccum = maccum_calc
          opt.projected_multiaccum = maccum_calc 
                                  
      ch.detector_readout.exposure_time.val = (t_int + dead_time + zero_time) 
      
      jexosim_msg ("Estimated integration time - zeroth read %s"%(t_int), opt.diagnostics)  
      jexosim_msg ("Estimated integration time incl zeroth read %s"%(t_int  + zero_time), opt.diagnostics)
      jexosim_msg ("Estimated TOTAL CYCLE TIME %s"%(ch.detector_readout.exposure_time.val), opt.diagnostics)
      
      if ch.detector_readout.exposure_time.val < (2*ch.detector_readout.t_g.val +  dead_time):
          jexosim_msg ("not enough exposure time for 2 groups + dead time", opt.diagnostics)
          opt.observation_feasibility = 0          
      
      jexosim_msg ("CDS time estimate %s"%(t_int), opt.diagnostics)          
      jexosim_msg ("FP max %s"%(opt.fp[1::3,1::3].max()), opt.diagnostics)
      jexosim_msg ("DC SWITCH.......%s"%(opt.background.EnableDC.val), opt.diagnostics)
      jexosim_msg ("DISABLE ALL SWITCH.......%s"%(opt.background.DisableAll.val), opt.diagnostics)   
      jexosim_msg ("DARK CURRENT %s"%(ch.detector_pixel.Idc.val) , opt.diagnostics)
  
      jexosim_plot('focal plane check', opt.diagnostics, image=True, 
                  image_data=opt.fp_signal[1::3,1::3], aspect='auto', interpolation = None,
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

      if opt.noise.ApplyRandomPRNU.val == 1:
          opt.qe = np.random.normal(1, opt.noise.QErms.val, opt.fp[1::3,1::3].shape) # for random uncertainty
          opt.qe_uncert = np.random.normal(1, opt.noise.QErms_uncert.val, opt.fp[1::3,1::3].shape) # for random uncertainty  
          jexosim_msg ("RANDOM PRNU GRID SELECTED...",  opt.diagnostics)
      else:
          opt.qe = np.load('%s/data/JWST/PRNU/qe_rms.npy'%(opt.__path__))[0:opt.fp[1::3,1::3].shape[0],0:opt.fp[1::3,1::3].shape[1]]
          opt.qe_uncert = np.load('%s/data/JWST/PRNU/qe_uncert.npy'%(opt.__path__))[0:opt.fp[1::3,1::3].shape[0],0:opt.fp[1::3,1::3].shape[1]]      
          jexosim_msg ("PRNU GRID SELECTED FROM FILE...", opt.diagnostics)
         
      opt.fp_original = copy.deepcopy(opt.fp)
      opt.fp_signal_original = copy.deepcopy(opt.fp_signal)  
      opt.x_wav_osr_original = copy.deepcopy(opt.x_wav_osr)
      opt.x_pix_osr_original = copy.deepcopy(opt.x_pix_osr)  
      
      opt.zodi_sed_original = copy.deepcopy(opt.zodi.sed) # needed here due to possible cropping above for subarrays
      opt.emission_sed_original = copy.deepcopy(opt.emission.sed)
      opt.qe_original = copy.deepcopy(opt.qe)
      opt.qe_uncert_original = copy.deepcopy(opt.qe_uncert)
      
      jexosim_plot('final wl solution on subarray', opt.diagnostics,
                   ydata=opt.x_wav_osr[1::3],
                   xlabel = 'x \'spectral\' pixel', ylabel = 'y \'spatial\' pixel',
                   grid=True)
        
      
      sanity_check(opt)
        
      return opt
      
      
def sanity_check(opt):
    import scipy.constants as spc
    
    wl = opt.x_wav_osr[1::3]
    del_wl = -np.gradient(wl)
#    del_wl = opt.d_x_wav_osr[1::3]*3
    star_spec = opt.star_sed
    star_spec.rebin(wl)
    T = opt.planet.planet.star.T
    trans_sed = opt.total_transmission.sed*u.dimensionless_unscaled
    trans = Sed(opt.total_transmission.wl,trans_sed)
    trans.rebin(wl)
    QE = opt.qe_spec
    QE.rebin(wl)
    Rs = (opt.planet.planet.star.R).to(u.m)
    D = (opt.planet.planet.star.d).to(u.m)
    n= trans.sed*del_wl*np.pi*planck(wl,T)*(Rs/D)**2*opt.Aeff*QE.sed/(spc.h*spc.c/(wl*1e-6))
    n2= trans.sed*del_wl*star_spec.sed*opt.Aeff*QE.sed/(spc.h*spc.c/(wl*1e-6))
    
    jex_sig = opt.fp_signal[1::3,1::3].sum(axis=0)
    
    R = opt.channel.data_pipeline.R.val
    del_wav = wl/R
    opt.exp_sig  = opt.t_int*del_wav*jex_sig/del_wl
    
    if opt.diagnostics ==1:
        plt.figure('sanity check 1 - check focal plane signal')
        plt.plot(wl,n, 'bo', label='BB check')
        plt.plot(wl,n2, 'ro', label='Phoenix check')  # not convolved with PSF unlike JexoSim, so peak may be higher
        plt.plot(wl, jex_sig, 'gx', label='JexoSim')
        plt.ylabel('e/s/pixel col'); plt.xlabel('pixel col wavelength (microns)')
        plt.legend(loc='best')
        
        plt.figure('sanity check 2 - expected final star signal in R bin of %s'%((R)))
        plt.plot(wl, opt.exp_sig)
        plt.ylabel('e/bin'); plt.xlabel('Wavelength (microns)')
            
        plt.figure('sanity check 3 - expected photon noise (sd) in R bin of %s'%((R)))
        plt.plot(wl, opt.exp_sig**0.5)    
        plt.ylabel('e/bin'); plt.xlabel('Wavelength (microns)')    

