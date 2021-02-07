"""
JexoSim 
2.0
Detector module
v1.0

"""
from jexosim.lib import instrument_lib
from jexosim.lib.jexosim_lib import jexosim_msg, jexosim_plot
import numpy           as np
import matplotlib.pyplot as plt
import copy
 
def run(opt):
    
      opt.observation_feasibility = 1  # this variable is currently not changed to zero under any circumstances
      opt.psf, opt.psf_type  = instrument_lib.get_psf(opt)
      opt.fp, opt.fp_signal = instrument_lib.get_focal_plane(opt)
      # test_fp = instrument_lib.psf_fp_check(opt)  # this can be commented out : checks that PSFs contribute no additional signal 
      opt.planet.sed = instrument_lib.get_planet_spectrum(opt)
      opt.fp, opt.fp_signal = instrument_lib.convolve_prf(opt) #this step can gen some small negative values on fp
      opt.fp = np.where(opt.fp<0,0, opt.fp)
      opt.fp_signal = np.where(opt.fp_signal<0,0, opt.fp_signal)
    
      opt = instrument_lib.user_subarray(opt) 
      opt = instrument_lib.crop_to_subarray(opt)
      opt = instrument_lib.exposure_timing(opt)

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
      opt.qy_zodi_original = copy.deepcopy(opt.qy_zodi)
      opt.qy_emission_original = copy.deepcopy(opt.qy_emission)

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

      instrument_lib.sanity_check(opt)
      
   
      return opt
