"""
JexoSim 2.0

Instrument library

"""
import numpy as np
from jexosim.lib import jexosim_lib
from jexosim.lib.jexosim_lib import jexosim_msg, jexosim_plot, planck
from astropy import units as u
from astropy import constants as const
from jexosim.classes.sed import Sed
import scipy
import matplotlib.pyplot as plt
import sys, os
from scipy import interpolate, signal
import copy

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

      # pos = (dtmp[...,2]*u.um  + offset.to(u.um)) + (opt.channel.camera.dispersion() ).to(u.um)   
      pos = (dtmp[...,2]*u.um  + offset.to(u.um))  
      
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
      idx = np.argwhere(x_wav_osr.value<wav.min())
      x_wav_osr[idx] = 0*u.um 
      idx = np.argwhere(x_wav_osr.value>wav.max())
      x_wav_osr[idx] = 0*u.um               
      idx = np.argwhere(x_pix_osr.value < pos.value.min())
      x_wav_osr[idx] = 0*u.um
      idx = np.argwhere(x_pix_osr.value > pos.value.max())
      x_wav_osr[idx] = 0*u.um                   
      if pos_y[0] !=0:
# =============================================================================
#           This method does not work properly - thus use the interpolation method for y_pos_osr
          # y_pos_osr =0
          # for i in range (0,r_y+1):
          #     y_pos_osr = y_pos_osr + z_y[i]*x_wav_osr.value**(r_y-i)        
          # y_pos_osr = (y_pos_osr/((opt.fp_delta).to(u.um))).value
# =============================================================================
          y_pos_osr = scipy.interpolate.interp1d(dtmp[...,0],  dtmp[...,1], bounds_error=False, fill_value=0.0)(x_wav_osr)
          # there must be a fill value as each x_wav_osr must have a corresponding y_pos_osr
          y_pos_osr = (y_pos_osr/((opt.fp_delta).to(u.um))).value  
      else:
          y_pos_osr = []         
      return x_wav_osr, x_pix_osr, y_pos_osr
            
def getR(opt): #this needs updating
      disp = opt.channel.camera.dispersion.path.replace(
	  '__path__', opt.__path__) 
      R_file = disp.replace('dispersion', 'R') 
      dtmp=np.loadtxt(R_file, delimiter=',') 
      wav = dtmp[...,0]*u.um
      R = dtmp[...,1]*u.dimensionless_unscaled 
      R = np.interp(opt.x_wav_osr, wav, R, left=None, right=None)
      R = Sed(opt.x_wav_osr,R)
      return R
      
def sanity_check(opt):
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
   
    n= quantum_yield.sed* trans.sed*del_wl*np.pi*planck(wl,T)*(Rs/D)**2*opt.Aeff*QE.sed/(const.h.value*const.c.value/(wl*1e-6))
    n2= quantum_yield.sed* trans.sed*del_wl*star_spec.sed*opt.Aeff*QE.sed/(const.h.value*const.c.value/(wl*1e-6))    
    jex_sig = quantum_yield.sed*opt.fp_signal[1::3,1::3].sum(axis=0)
    
    n = np.where(n<0,0,n)
    n2 = np.where(n2<0,0,n2)
    
    R = opt.pipeline.pipeline_R.val
    del_wav = wl/R
    opt.exp_sig  = opt.t_int*del_wav*jex_sig/del_wl
    if opt.diagnostics ==1:
        plt.figure('sanity check 1 - check focal plane signal')
        if 'NIRISS' in opt.channel.name:
            plt.plot(wl[:-10],n[:-10], 'b^', label='BB check')
            plt.plot(wl[:-10],n2[:-10], 'r+', label='Phoenix check')  # not convolved with PSF unlike JexoSim, so peak may be higher
        else:
            plt.plot(wl,n, 'b^', label='BB check')
            plt.plot(wl,n2, 'r+', label='Phoenix check')  # not convolved with PSF unlike JexoSim, so peak may be higher
        plt.plot(wl, jex_sig, 'gx', label='JexoSim')
        plt.ylabel('e/s/pixel col'); plt.xlabel('pixel col wavelength (microns)')
        plt.legend(loc='best')
       
     ################      
        plt.figure('sanity check 2 - expected final star signal in R bin of %s'%((R)))
        plt.plot(wl, opt.exp_sig)
        plt.ylabel('e/bin'); plt.xlabel('Wavelength (microns)')
      ################      
        plt.figure('sanity check 3 - expected photon noise (sd) in R bin of %s'%((R)))
        plt.plot(wl, opt.exp_sig**0.5)    
        plt.ylabel('e/bin'); plt.xlabel('Wavelength (microns)')     

def user_subarray(opt):    
    # s = (opt.observation.obs_inst_config.val).replace(" ", "")
    # start = 0
    # idx=[]
    # for i in range(3):
    #      idx0 = s.find('+', start)
    #      idx.append(idx0)
    #      start =idx0+1
    # opt.subarray = s[idx[1]+1:idx[2]]
    # opt.pattern = s[idx[2]+1:]
    
   
    aa = opt.observation.obs_inst_config.val.split('+') 
    for i in range(len(aa)):
        aa[i] = aa[i].strip()
    opt.subarray =  aa[2]   
    opt.pattern = aa[3]
    if opt.channel.instrument.val =='NIRCam':
        bb = opt.subarray.split('_')
        opt.subarray_name = bb[0]
        opt.subarray_output_number = bb[1]
    else:
        opt.subarray_name = opt.subarray
   
    
################
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
    if opt.fp_y == opt.fpn[0] and  opt.fp_x == opt.fpn[1]:    # do not crop as already right size     
            pass
    else: # set new subarray size for fp and fp_signal
        jexosim_msg ("now cropping to chosen subarray", opt.diagnostics)
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
            
            
        print ('xcrop', 'xcrop0', xcrop, xcrop0)
        if xcrop is not None:
            if xcrop%3 != 0:
                print ('error: xcrop must be multiple of 3')
                xxx
        if xcrop0 is not None:
            if xcrop0%3 != 0:
                print ('error: xcrop0 must be multiple of 3')
                xxx
        opt.fp = opt.fp[ycrop:ycrop0]
        opt.fp = opt.fp[:,xcrop:xcrop0]      
        opt.fp_signal = opt.fp_signal[ycrop:ycrop0]
        opt.fp_signal = opt.fp_signal[:,xcrop:xcrop0]        
        opt.x_wav_osr = opt.x_wav_osr[xcrop:xcrop0]
        opt.d_x_wav_osr = opt.d_x_wav_osr[xcrop:xcrop0]
        opt.x_pix_osr = opt.x_pix_osr[xcrop:xcrop0]
        opt.zodi.sed = opt.zodi.sed[xcrop:xcrop0]
        opt.sunshield.sed = opt.sunshield.sed[xcrop:xcrop0]
        opt.emission.sed = opt.emission.sed[xcrop:xcrop0]
        opt.zodi.wl = opt.x_wav_osr
        opt.sunshield.wl = opt.x_wav_osr
        opt.emission.wl = opt.x_wav_osr
        if xcrop0 is None:
            opt.planet.sed.sed =opt.planet.sed.sed[int(xcrop/3):]                  
            opt.planet.sed.wl = opt.planet.sed.wl[int(xcrop/3):] 
        else: 
            opt.planet.sed.sed =opt.planet.sed.sed[int(xcrop/3):int(xcrop0/3)]                  
            opt.planet.sed.wl = opt.planet.sed.wl[int(xcrop/3):int(xcrop0/3)]
        if len(opt.planet.sed.wl) != len(opt.x_wav_osr)/3:
            print ('error: size of planet spectrum not matching array width')
            xxx
                
        opt.R.sed = opt.R.sed[xcrop:xcrop0] 
        opt.R.wl = opt.x_wav_osr
        opt.quantum_yield.sed = opt.quantum_yield.sed[xcrop:xcrop0] 
        opt.quantum_yield.wl = opt.x_wav_osr 
        opt.qy_zodi = opt.qy_zodi[xcrop:xcrop0] 
        opt.qy_sunshield = opt.qy_sunshield[xcrop:xcrop0] 
        opt.qy_emission = opt.qy_emission[xcrop:xcrop0] 
 
        
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

def get_psf(opt):
      if os.path.exists('%s/../archive/PSF/%s_psf_stack.npy'%(opt.__path__,opt.channel.instrument.val)):
          psf_stack = np.load('%s/../archive/PSF/%s_psf_stack.npy'%(opt.__path__,opt.channel.instrument.val))
          psf_stack_wl =  1e6*np.load('%s/../archive/PSF/%s_psf_stack_wl.npy'%(opt.__path__,opt.channel.instrument.val))
          psf = interpolate.interp1d(psf_stack_wl, psf_stack, axis=2,bounds_error=False, fill_value=0.0, kind='linear')(opt.x_wav_osr.value)      
          psf = np.rot90(psf) 
          psf_type = 'wfe'   
      else: # uses airy if no psf database however this is to be avoided, as pipeline assumes the wfe database psfs.
          jexosim_msg('PSF files could not be found. Check psf files location.')
          sys.exit()
          # psf = jexosim_lib.Psf(opt.x_wav_osr.value, opt.channel.camera.wfno_x.val, opt.channel.camera.wfno_y.val, opt.fp_delta.value, shape='airy')  
          # psf[np.isnan(psf)] =0
          # psf_type = 'airy'
           
      jexosim_msg("PSF shape %s, %s"%(psf.shape[0],psf.shape[1] ), opt.diagnostics)     
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
      return psf, psf_type         
  
def get_focal_plane(opt):  
      # Populate focal plane with monochromatic PSFs
      j0 = np.arange(opt.fp.shape[1]) - int(opt.psf.shape[1]/2)
      j1 = j0 + opt.psf.shape[1]
      idx = np.where((j0>=0) & (j1 < opt.fp.shape[1]))[0]
      i0 = np.array([opt.fp.shape[0]/2 - opt.psf.shape[0]/2 + opt.offs]*len(j1)).astype(np.int)     
      i0+=1         
      # variable y position (applies only to NIRISS)
      if opt.y_pos_osr != []:
          y_pos_osr0 = np.where(opt.y_pos_osr<0,0, opt.y_pos_osr).astype(np.int) +opt.fp.shape[0]/2 -opt.psf.shape[0]/2 + opt.offs 
          i0 = (y_pos_osr0).astype(np.int) 
      i1 = i0 + opt.psf.shape[0]  
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
              opt.fp[i0[k]:i1[k], j0[k]:j1[k]] += opt.psf[...,k] * opt.star.sed.sed[k].value              
      for k in idx: # used for getting sat time, and sizing
          opt.fp_signal[i0[k]:i1[k], j0[k]:j1[k]] += opt.psf[...,k] * opt.star.sed.sed[k].value                        
      # SPECIAL CASE: fix if airy psfs used      
      if opt.channel.name == 'NIRISS_SOSS_GR700XD': 
          if opt.psf_type == 'airy':  # now return fp to original size
              if i1.max() > original_fp.shape[0]: 
                  diff = i1.max()- original_fp.shape[0]         
                  opt.fp = opt.fp[diff:]    
                  opt.fp_signal = opt.fp_signal[diff:]  
 
      return opt.fp, opt.fp_signal
    
    
def get_focal_plane_it(opt):  
      # Populate focal plane with monochromatic PSFs
      j0 = np.arange(opt.fp_it.shape[1]) - int(opt.psf.shape[1]/2)
      j1 = j0 + opt.psf.shape[1]
      idx = np.where((j0>=0) & (j1 < opt.fp_it.shape[1]))[0]
      i0 = np.array([opt.fp_it.shape[0]/2 - opt.psf.shape[0]/2 + opt.offs]*len(j1)).astype(np.int)     
      i0+=1         
      # variable y position (applies only to NIRISS)
      if opt.y_pos_osr != []:
          y_pos_osr0 = np.where(opt.y_pos_osr<0,0, opt.y_pos_osr).astype(np.int) +opt.fp_it.shape[0]/2 -opt.psf.shape[0]/2 + opt.offs 
          i0 = (y_pos_osr0).astype(np.int) 
      i1 = i0 + opt.psf.shape[0]  
      # SPECIAL CASE: fix if airy psfs used     
      if opt.channel.name == 'NIRISS_SOSS_GR700XD':
          # fix if airy psfs used
          if opt.psf_type == 'airy':
              original_fp_it = copy.deepcopy(opt.fp_it)
              if i1.max() > opt.fp_it.shape[0]: #psfs will fall outside fp_it area due to curve
                  original_fp_it = copy.deepcopy(opt.fp_it)
                  opt.fp_it = np.zeros((i1.max(), opt.fp_it.shape[1] ))  # temp increase in fp_it size  
                  opt.fp_signal_it = np.zeros((i1.max(), opt.fp_signal_it.shape[1] ))  # temp increase in fp_it size           
      if opt.background.EnableSource.val == 1 or opt.background.EnableAll.val == 1:     
          for k in idx: # actual signal 
              opt.fp_it[i0[k]:i1[k], j0[k]:j1[k]] += opt.psf[...,k] * opt.star.sed_it.sed[k].value              
      for k in idx: # used for getting sat time, and sizing
          opt.fp_signal_it[i0[k]:i1[k], j0[k]:j1[k]] += opt.psf[...,k] * opt.star.sed_it.sed[k].value                        
      # SPECIAL CASE: fix if airy psfs used      
      if opt.channel.name == 'NIRISS_SOSS_GR700XD': 
          if opt.psf_type == 'airy':  # now return fp_it to original size
              if i1.max() > original_fp_it.shape[0]: 
                  diff = i1.max()- original_fp_it.shape[0]         
                  opt.fp_it = opt.fp_it[diff:]    
                  opt.fp_signal_it = opt.fp_signal_it[diff:]                   
      return opt.fp_it, opt.fp_signal_it


def get_planet_spectrum(opt): 
    fp_signal_it = opt.fp_signal_it[1::3, 1::3]
    fp_signal = opt.fp_signal[1::3, 1::3]
    
    rat = (1- (fp_signal_it/fp_signal))
    rat[np.isnan(rat)] = 0
    cr_average = np.zeros(fp_signal.shape[1])
    for i in range(fp_signal.shape[1]):
        slice = rat[:,i]
        idx = np.argwhere(slice>0)
        slice = slice[idx]
        # wt = 1/slice
        # this is important to remove outliers
        idx = np.argwhere(abs(slice-np.median(slice)) > 0.01*np.median(slice)).T[0]
        # sd = np.std(slice)
        # idx = np.argwhere(abs(slice-np.mean(slice) > sd))    
        slice = np.delete(slice, idx)
        # wt = 1/slice
        # if i == 200:
        #     print (idx, 0.1*np.median(slice))
        #     plt.plot(slice, 'ro-')  
        # # if i == 1760:
        #     print (idx, 0.1*np.median(slice))
        #     plt.plot(slice, 'bo-')
        cr_average[i] = np.mean(slice)
        # cr_average[i] = np.sum(slice*wt)/ np.sum(wt)

    opt.planet.sed =  Sed(opt.x_wav_osr[1::3], cr_average*u.dimensionless_unscaled)

    # remove values outside of the wavelength range of the channel
    idx = np.argwhere(opt.planet.sed.wl.value < opt.channel.pipeline_params.wavrange_lo.val-0.5).T[0]
    opt.planet.sed.sed[idx] = 0*u.dimensionless_unscaled
    idx = np.argwhere(opt.planet.sed.wl.value > opt.channel.pipeline_params.wavrange_hi.val+0.5).T[0]
    opt.planet.sed.sed[idx] = 0*u.dimensionless_unscaled
    idx = np.argwhere(opt.planet.sed.wl.value ==0).T[0]
    opt.planet.sed.sed[idx] = 0*u.dimensionless_unscaled
 
    return opt.planet.sed
 
     

def convolve_prf(opt, fp , fp_signal):
     if opt.channel.detector_pixel.pixel_diffusion_length.val.value < 0.01:
         box = np.ones((3,3))
         fp =  signal.fftconvolve(fp, box, 'same') 
         fp_signal =  signal.fftconvolve(fp_signal, box, 'same') 
         jexosim_msg('Convolving with top-hat PRF', opt.diagnostics)
     
     else:
          kernel, kernel_delta = jexosim_lib.PixelResponseFunction(opt, 
            opt.psf.shape[0:2],
            7*opt.channel.simulation_factors.osf(),   
            opt.channel.detector_pixel.pixel_size(),
            lx = opt.channel.detector_pixel.pixel_diffusion_length())
          # jexosim_msg ("kernel sum %s"%(kernel.sum()), opt.diagnostics) 
          # jexosim_msg ("check 3.9 - unconvolved FP max %s"%(fp.max()) , opt.diagnostics)      
          # jexosim_plot('test3', opt.diagnostics, xdata=opt.x_wav_osr, ydata = fp.sum(axis=0), marker='bo')
          # jexosim_plot('test4', opt.diagnostics, xdata=opt.x_pix_osr, ydata = opt.x_wav_osr, marker='bo')                  
          # jexosim_plot('test5', opt.diagnostics, xdata=opt.x_pix_osr, ydata = fp.sum(axis=0), marker='bo')
          # jexosim_plot('test6', opt.diagnostics, xdata=opt.x_wav_osr, ydata = opt.star.sed.sed, marker='bo') 
          fp = jexosim_lib.fast_convolution(fp, opt.fp_delta, kernel, kernel_delta)
          fp_signal = jexosim_lib.fast_convolution(fp_signal, opt.fp_delta, kernel, kernel_delta)
          jexosim_msg ("check 4 - convolved FP max %s"%(fp.max()), opt.diagnostics)  
          jexosim_msg ("check 4 - convolved FP signal max %s"%(fp_signal.max()), opt.diagnostics)    
    
          opt.kernel = kernel
          opt.kernel_delta = kernel_delta     
          # Fix units
     fp = fp*opt.star.sed.sed.unit 
     fp_signal = fp_signal*opt.star.sed.sed.unit  
     return fp, fp_signal
  
def exposure_timing(opt):    
    
    ## Find count rate with diffuse radiation
      fp_copy = copy.deepcopy(opt.fp_signal)
      fp_count_no_bkg = 1*fp_copy[1::3,1::3] * opt.quantum_yield.sed[1::3]
      fp_count = opt.zodi.sed[1::3]*opt.qy_zodi[1::3] + opt.sunshield.sed[1::3]*opt.qy_sunshield[1::3]  + opt.emission.sed[1::3]*opt.qy_emission[1::3] + fp_copy[1::3,1::3] * opt.quantum_yield.sed[1::3]      
      fp_count += opt.channel.detector_pixel.Idc.val 
      
      jexosim_msg ("check 4a - %s"%(fp_count.max()), opt.diagnostics)

      fp_count = fp_count.value
      fp_count_no_bkg = fp_count_no_bkg.value 
      if opt.simulation.sim_use_ipc.val == 1 and opt.channel.instrument.val !='MIRI':
          ipc_kernel = np.load(opt.channel.detector_array.ipc_kernel.val.replace('__path__', '%s/%s'%(opt.jexosim_path, 'jexosim')))
          fp_count = scipy.signal.fftconvolve(fp_count, ipc_kernel, 'same') 
          fp_count_no_bkg = scipy.signal.fftconvolve(fp_count_no_bkg, ipc_kernel, 'same')
          # this will reduce the max count a bit.  ipc kernel not applied to images until noise module    
    ### apply gap for these modes to fp_count and fp_count_no_bkg for sat time measurement
      if opt.channel.name == 'NIRSpec_BOTS_G140H_F100LP' or \
        opt.channel.name == 'NIRSpec_BOTS_G235H_F170LP'\
            or opt.channel.name == 'NIRSpec_BOTS_G395H_F290LP':      
          idx0 = np.argwhere(opt.x_wav_osr[1::3] ==opt.wav_gap_start)[0].item() #recover the start of the gap
          idx1 = idx0 + opt.gap_len # gap of 172 pix works well for G1H and G2H, but a little off (by 0.01 microns for the other in some cases from published values)
          fp_count[:, idx0:idx1] = 0
          fp_count_no_bkg[:, idx0:idx1] = 0 
          plt.figure('high res grating exposure fpa check')
          plt.imshow(fp_count)
          plt.figure('high res grating exposure fpa wav check')
          plt.plot(opt.x_wav_osr[1::3], fp_count.sum(axis=0))
      jexosim_msg ("check 5 - %s"%(fp_count.max()), opt.diagnostics)
      FW = opt.channel.detector_pixel.full_well.val       
      A,B = np.unravel_index(fp_count.argmax(), fp_count.shape)
      jexosim_msg ("maximum index and count with all backgrounds %s %s %s"%(A,B, fp_count.max()), opt.diagnostics)      
      A,B = np.unravel_index(fp_count_no_bkg.argmax(), fp_count_no_bkg.shape)
      jexosim_msg ("maximum index and count with no backgrounds %s %s %s"%(A,B, fp_count_no_bkg.max()), opt.diagnostics)   
      jexosim_msg ("full well in electron %s"%(FW), opt.diagnostics)   
      jexosim_msg ("saturation time assuming 100 percent full well with all backgrounds %s"%((FW / fp_count.max())), opt.diagnostics)     
      jexosim_msg ("full well percentage chosen for saturation limit: %s"%(opt.observation.obs_fw_percent.val), opt.diagnostics)
      opt.sat_time = ((FW / fp_count.max()) *opt.observation.obs_fw_percent.val/100.0 )*u.s     
      opt.sat_time_no_bkg = ((FW / fp_count_no_bkg.max()) *opt.observation.obs_fw_percent.val/100.0 )*u.s
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
    ######  Set effective multiaccum
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
      return opt
    
    
def psf_fp_check(opt):
       # Populate focal plane with monochromatic PSFs
      j0 = np.arange(opt.fp.shape[1]) - int(opt.psf.shape[1]/2)
      j1 = j0 + opt.psf.shape[1]
      idx = np.where((j0>=0) & (j1 < opt.fp.shape[1]))[0]
      i0 = np.array([opt.fp.shape[0]/2 - opt.psf.shape[0]/2 + opt.offs]*len(j1)).astype(np.int)     
      i0+=1         
      # variable y position (applies only to NIRISS)
      if opt.channel.name == 'NIRISS_SOSS_GR700XD':
          i0 = (opt.y_pos_osr).astype(np.int) 
      i1 = i0 + opt.psf.shape[0]  
      test_fp = np.zeros_like(opt.fp)
      
      # SPECIAL CASE: fix if airy psfs used   
      if opt.channel.name == 'NIRISS_SOSS_GR700XD':
          # fix if airy psfs used
          if opt.psf_type == 'airy':
              original_fp = copy.deepcopy(test_fp)
              if i1.max() > test_fp.shape[0]: #psfs will fall outside fp area due to curve
                  original_fp = copy.deepcopy(test_fp)
                  test_fp = np.zeros((i1.max(), test_fp.shape[1] ))  # temp increase in fp size          
          
      for k in idx: # used for getting sat time, and sizing
          test_fp[i0[k]:i1[k], j0[k]:j1[k]] += opt.psf[...,k]                        
      # SPECIAL CASE: fix if airy psfs used      
      if opt.channel.name == 'NIRISS_SOSS_GR700XD': 
          if opt.psf_type == 'airy':  # now return fp to original size
              if i1.max() > original_fp.shape[0]: 
                  diff = i1.max()- original_fp.shape[0]         
                  test_fp = test_fp[diff:]   
      
      jexosim_plot('psf fp check', 1, ydata = test_fp.sum(axis=0))
       
      return test_fp 
      