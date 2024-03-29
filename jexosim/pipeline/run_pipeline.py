"""
JexoSim 
2.0
Pipeline run module
v1.0

"""

import numpy as np
from jexosim.pipeline import calibration, jitterdecorr, binning
from jexosim.lib.jexosim_lib import jexosim_msg, jexosim_plot
from astropy import units as u  
    
class pipeline_stage_1():
       
    def __init__(self, opt):
        
        self.opt = opt
        
        self.exp_end_time_grid = self.opt.ndr_end_time[self.opt.effective_multiaccum-1::self.opt.effective_multiaccum]
        self.ndr_end_time_grid = self.opt.ndr_end_time
        self.opt.data_raw = opt.data*1
        
        self.ApFactor = opt.pipeline.pipeline_ap_factor.val

        if opt.background.EnableSource.val == 1:
            self.opt.diff =0
        else:
            self.opt.diff =1
           
        self.loadData()  
        opt.init_pix = np.zeros(opt.data[...,0].shape) # placeholder - replace with initial bad pixel map 
        
        self.dqInit()  
        
        self.satFlag() 
                      
        if (opt.background.EnableDC.val ==1 or opt.background.EnableAll.val == 1) and opt.background.DisableAll.val != 1: 
            if opt.diff==0:
                self.subDark()             
                
        if (opt.noise.ApplyPRNU.val ==1 or opt.noise.EnableAll.val == 1) and opt.noise.DisableAll.val != 1: 
            jexosim_msg ("mean and standard deviation of qe grid %s %s"%(opt.qe_grid.mean(), opt.qe_grid.std()), opt.diagnostics)
               
            self.flatField()
            
        if (opt.background.EnableZodi.val ==1 or opt.background.EnableSunshield.val ==1 or opt.background.EnableEmission.val ==1 or opt.background.EnableAll.val == 1) and opt.background.DisableAll.val != 1: 
            if opt.diff==0:                  
                self.subBackground()
        jexosim_plot('sample NDR image', opt.diagnostics,
                     image =True, image_data=self.opt.data[...,1])
       
        self.doUTR() # lose astropy units at this step
      
        jexosim_plot('sample exposure image', opt.diagnostics,
                     image =True, image_data=self.opt.data[...,1]) 
        
        
        
         # currently this applies zero values to saturated pixels
        self.badCorr()    # exp image obtained here     
   
        if (opt.noise.EnableSpatialJitter.val  ==1 or opt.noise.EnableSpectralJitter.val  ==1  or opt.noise.EnableAll.val ==1) and opt.noise.DisableAll.val != 1:
            jexosim_msg ("Decorrelating pointing jitter...", opt.diagnostics)
            self.jitterDecorr()
            
        if opt.pipeline.pipeline_apply_mask.val==1:
            if opt.pipeline.pipeline_auto_ap.val == 1: 
                self.autoSizeAp()            
            
        self.extractSpec()
              

#==============================================================================
#    Pipeline steps       
#             
#==============================================================================
         
    def loadData(self):
        self.opt = calibration.loadData(self.opt)
        
    def dqInit(self):   
        self.dq_array = calibration.dqInit(self.opt.data, self.opt)        
        
    def satFlag(self):   
        self.dq_array = calibration.satFlag(self.opt.data, self.opt)        
          
    def subZero(self):   
        self.opt = calibration.subZero(self.opt)
        
    def subDark(self):
        self.opt = calibration.subDark(self.opt)

    def flatField(self):
        self.opt = calibration.flatField(self.opt)
   
    def subBackground(self):


        self.opt = calibration.subBackground(self.opt)  
        
    def doUTR(self):
        self.opt.data = calibration.doUTR(self.opt.data, self.opt) 
        
        if self.opt.pipeline.useSignal.val == 1:
            self.opt.data_signal_only = calibration.doUTR(self.opt.data_signal_only, self.opt)
            
    def badCorr(self):   
        self.opt = calibration.badCorr(self.opt.data, self.opt)              
                    
    def jitterDecorr(self):  
         
        self.jitterCode = jitterdecorr.jitterCode(self.opt.data, self.opt)
        self.opt.data = self.jitterCode.decorrData
        
    def autoSizeAp(self):     
        F = self.opt.channel.camera.wfno_x.val
        wl_max = self.opt.channel.pipeline_params.wavrange_hi.val 
        wl_min = self.opt.channel.pipeline_params.wavrange_lo.val
        wl = self.opt.cr_wl.value   
        if self.opt.timeline.apply_lc.val ==0 :       
            x=100
            if self.opt.data.shape[2] <x: # has to use the noisy data
                x= self.opt.data.shape[2]
            sample = self.opt.data[...,0:x]
        
        elif self.opt.timeline.apply_lc.val ==1 :     # must use OOT portions else the transit will contribute falsely to the stand dev
            if self.opt.pipeline.split ==0: # can only use the pre-transit values     
                total_obs_time =  self.opt.exposure_time*self.opt.n_exp
                pre_transit_time= self.opt.observation.obs_frac_t14_pre_transit.val * self.opt.T14   
                post_transit_time= self.opt.observation.obs_frac_t14_post_transit.val * self.opt.T14   
                pre_transit_exp = self.opt.n_exp*pre_transit_time/total_obs_time
                post_transit_exp = self.opt.n_exp*post_transit_time/total_obs_time
                x0 = int(pre_transit_exp*0.75)
                if x0 >20:
                    x0=20
                x1 = int(post_transit_exp*0.75)
                if x1 >20:
                    x1=20 
                sample1 =  self.opt.data[...,0:x0]
                sample2 =  self.opt.data[...,-x1:]
                sample = np.dstack((sample1,sample2))   
             
            if self.opt.pipeline.split ==1: # can only use the pre-transit values
                total_obs_time =  self.opt.exposure_time*self.opt.n_exp
                pre_transit_time= self.opt.observation.obs_frac_t14_pre_transit.val * self.opt.T14   
                pre_transit_exp = pre_transit_time/self.opt.exposure_time
                if self.opt.n_exp < pre_transit_exp:
                    x0 = int(self.opt.n_exp*0.75)
                else:
                    x0 = int(pre_transit_exp*0.75)
                if x0 > 40:
                    x0=40
                sample =  self.opt.data[...,0:x0]
        # print (sample.shape)   
        # zero_check = sample.sum(axis=0)
        # zero_check = zero_check.sum(axis=1)
        # idx_zero = np.argwhere(zero_check==0)
        
        # sample = np.delete(sample, idx_zero, axis=1)
        
        # print (sample.shape)
        # print (idx_zero)
        # idx_zero_0 = idx_zero[0].item()
        # idx_zero_1 = idx_zero[-1].item()
        
        # print (idx_zero_0, idx_zero_1)
        # print (zero_check[idx_zero_0], zero_check[idx_zero_1])
        
        # import matplotlib.pyplot as plt
        # plt.figure('zero_check')
        # plt.plot(zero_check)
        # xxxx
            
        jexosim_msg("SAMPLE SHAPE %s %s %s"%(sample.shape), self.opt.diagnostics)
        pix_size = (self.opt.channel.detector_pixel.pixel_size.val).to(u.um).value
        y_width = self.opt.data.shape[0] * pix_size
        testApFactorMax = int(int(y_width/2.) / (F*wl_max))
        if (testApFactorMax*F*wl_max/pix_size) + 1 >= self.opt.data.shape[0]/2:
            testApFactorMax = int( ((self.opt.data.shape[0]/2)-1)*pix_size/(F*wl_max) )

        ApList = np.arange(1.0,testApFactorMax+1,1.0)
        if self.opt.channel.instrument.val =='NIRISS':
            testApFactorMax = 18.0 # set empirically
            ApList = np.arange(7.0,testApFactorMax+1,1.0)            
        jexosim_msg ("maximum test ap factor %s"%(testApFactorMax), self.opt.diagnostics)
        for i in ApList:
            testApFactor = 1.0*i  
            jexosim_msg("test ApFactor %s"%(testApFactor), self.opt.diagnostics)
            self.extractSample = binning.extractSpec(sample,  self.opt, self.opt.diff, testApFactor, 2) 
            if self.opt.channel.instrument.val =='NIRISS':
                    self.extractSample.applyMask_extract_1D_NIRISS()
            else:
                self.extractSample.applyMask_extract_1D()
            spectra = self.extractSample.spectra
            self.extractSample.binSpectra()   
            binnedLC = self.extractSample.binnedLC   
            wl =  self.extractSample.binnedWav
            SNR = binnedLC.mean(axis =0)/binnedLC .std(axis =0)
 
            if i == ApList[0] :
                SNR_stack = SNR
            else:
                SNR_stack = np.vstack((SNR_stack, SNR))
            
            jexosim_plot('test aperture SNR', self.opt.diagnostics, 
                        xdata=wl,ydata=SNR, label = testApFactor)   
        
        idx = np.argwhere( (wl>=wl_min) & (wl<= wl_max))
        SNR_stack = SNR_stack[:,idx][...,0]
        wl=  wl[idx].T[0]
        wl0=wl
        
 
        nan_check = SNR_stack.sum(axis=0)
        nan_idx = np.argwhere(np.isnan(nan_check))
    
        SNR_stack = np.delete(SNR_stack, nan_idx, axis=1)
        wl0 =  np.delete(wl, nan_idx) 
 
        zero_check = SNR_stack.sum(axis=0)
        zero_idx = np.argwhere(zero_check==0)
        SNR_stack = np.delete(SNR_stack, zero_idx, axis=1)
        wl0 =  np.delete(wl0, zero_idx) 
 
        

        best=[]
        for i in range(SNR_stack.shape[1]):
            aa = SNR_stack[:,i]
#            jexosim_msg i, np.argmax(aa), ApList[np.argmax(aa)]
            best.append(ApList[np.argmax(aa)])          
        

        jexosim_plot('highest SNR aperture vs wavelength', self.opt.diagnostics,
                     xdata=wl0, ydata=best, marker='bo-')      
        
        AvBest = np.round(np.mean(best),0)
        jexosim_msg ("average best aperture factor %s"%(AvBest), self.opt.diagnostics)
        self.opt.AvBest = AvBest
        self.ApFactor = AvBest
        self.opt.pipeline.pipeline_ap_factor.val = self.ApFactor

 
    def extractSpec(self):
        
        jexosim_msg ("extracting 1 D spectra....", self.opt.diagnostics)
        #==============================================================================
        # 1) initialise class objects a) noisy data, b) noiseless data if used, c) extraction of n_pix per bin
        #==============================================================================
        
        self.extractSpec = binning.extractSpec(self.opt.data, self.opt, self.opt.diff, self.ApFactor,1)
        # 1 = final ap : means code knows this is the ap factor to use on noisy data and plots
        # final ap  0 = signal only or n_pix evals, so no figures for aperture on image plotted.
        if self.opt.pipeline.useSignal.val == 1:        
            self.extractSpec_signal = binning.extractSpec(self.opt.data_signal_only, self.opt, 0, self.ApFactor, 0) #diff always 0 for signal
        
        # use a set of 3 images with 1 value for all pixels to find out no of pixels per bin
        self.extractSpec_nPix = binning.extractSpec(self.opt.data[...,0:3]*0+1, self.opt, 1, self.ApFactor, 0)
         
        #==============================================================================
        # 2) apply mask and extract 1D spectrum if apply mask selected      # special case for NIRISS 
        #==============================================================================
            
        if self.opt.pipeline.pipeline_apply_mask.val==1:             
            
            # a) noisy data   
            jexosim_msg ("applying mask and extracting 1D spectrum from noisy data", self.opt.diagnostics)
            if self.opt.channel.instrument.val =='NIRISS':
                    self.extractSpec.applyMask_extract_1D_NIRISS()
            else:
                    self.extractSpec.applyMask_extract_1D()
           
            # b) noiseless data if selected            
            if self.opt.pipeline.useSignal.val == 1:
                jexosim_msg ("applying mask and extracting 1D spectrum from signal only data", self.opt.diagnostics)
                if self.opt.channel.instrument.val =='NIRISS':
                    self.extractSpec_signal.applyMask_extract_1D_NIRISS()
                else:
                    self.extractSpec_signal.applyMask_extract_1D()
           
            # c) sample data to find n_pix per bin
            jexosim_msg ("applying mask and extracting 1D spectrum to find n_pix per bin" , self.opt.diagnostics)          
            if self.opt.channel.instrument.val =='NIRISS':
                self.extractSpec_nPix.applyMask_extract_1D_NIRISS()
            else:
                jexosim_msg ("extracting 1 D spectra for pixel in bin test",  self.opt.diagnostics)
                self.extractSpec_nPix.applyMask_extract_1D() 

        #==============================================================================
        # 3) extract 1D spectrum only if no mask selected 
        #==============================================================================         
        else:
            
            # a) noisy data 
            jexosim_msg ("NOT applying mask and extracting 1D spectrum from noisy data", self.opt.diagnostics)
            self.extractSpec.extract1DSpectra()
            
            # b) noiseless data if selected
            if self.opt.pipeline.useSignal.val == 1:  
                jexosim_msg ("NOT applying mask and extracting 1D spectrum from signal only data", self.opt.diagnostics)
                self.extractSpec_signal.extract1DSpectra()            
            
            # c) sample data to find n_pix per bin  
            jexosim_msg ("NOT applying mask and extracting 1D spectrum to find n_pix per bin", self.opt.diagnostics)
            self.extractSpec_nPix.extract1DSpectra()          
            

        self.nPix_1 =  self.extractSpec_nPix.spectra[0] 
        jexosim_plot('n_pix per pixel column', self.opt.diagnostics, 
                     xdata=self.opt.cr_wl.value, ydata=self.nPix_1, marker='bo')


        #==============================================================================
        # 4) Now bin into spectral bins
        #==============================================================================
        # a) noisy data 
        jexosim_msg ("binning 1D spectra into spectral bins... from noisy data", self.opt.diagnostics)
        self.extractSpec.binSpectra()    

        # b) noiseless data if selected     
        if self.opt.pipeline.useSignal.val == 1: 
            jexosim_msg ("binning 1D spectra into spectral bins... from signal only data", self.opt.diagnostics)
            self.extractSpec_signal.binSpectra() 
            
        # c) sample data to find n_pix per bin    
        jexosim_msg ("binning 1D spectra into spectral bins... to find n_pix per bin", self.opt.diagnostics)
        self.extractSpec_nPix.binSpectra() 

        #==============================================================================
        # 5) Define objects from binning process
        #==============================================================================              
        self.binnedLC =  self.extractSpec.binnedLC  
        self.binnedWav =  self.extractSpec.binnedWav 
        
        if self.opt.pipeline.useSignal.val == 1:           
            self.binnedLC_signal =  self.extractSpec_signal.binnedLC    
            self.binnedWav_signal =  self.extractSpec_signal.binnedWav  
        else:
            self.binnedLC_signal =  self.binnedLC 
            self.binnedWav_signal = self.binnedWav        
 
        if self.opt.timeline.apply_lc.val ==1 :
            self.extractSpec.binGamma()  
            self.binnedGamma =  self.extractSpec.binnedGamma  
  
        self.nPix_2 =  self.extractSpec_nPix.binnedLC[0]
        
        jexosim_plot('n_pix per bin', self.opt.diagnostics,
                     xdata=self.binnedWav, ydata =self.nPix_2, marker= 'ro')
  

class pipeline_stage_2():
       
    def __init__(self, opt):
        
        self.opt  = opt
        self.binnedWav = opt.pipeline_stage_1.binnedWav
        self.binnedLC = opt.pipeline_stage_1.binnedLC
        self.binnedLC_signal = opt.pipeline_stage_1.binnedLC_signal
        if self.opt.timeline.apply_lc.val ==1 :
            self.binnedGamma = opt.pipeline_stage_1.binnedGamma
        self.nPix_2 =  opt.pipeline_stage_1.nPix_2
        self.nPix_1 =  opt.pipeline_stage_1.nPix_1
        self.ApFactor=  opt.pipeline_stage_1.ApFactor
        self.exp_end_time_grid = opt.ndr_end_time[opt.effective_multiaccum-1::opt.effective_multiaccum]
        self.ndr_end_time_grid = opt.ndr_end_time
        self.data_raw = opt.data_raw
        
        if self.opt.timeline.apply_lc.val ==1 :
            self.fitLC()            
        elif self.opt.timeline.apply_lc.val ==0 :
            self.ootSNR()
            
    def fitLC(self):
        self.processLC = binning.processLC(self.binnedLC, self.binnedWav, self.binnedGamma, self.opt)   
        self.transitDepths = self.processLC.transitDepths
        self.model_gamma1 = self.processLC.model_gamma1
        self.model_gamma2 = self.processLC.model_gamma2
        self.model_f = self.processLC.model_f
        self.binnedWav = self.processLC.binnedWav


    def ootSNR(self):
        if self.opt.pipeline.useSignal.val == 1:
            self.processOOT = binning.processOOT(self.binnedLC, self.binnedLC_signal,self.binnedWav, self.opt)   
        else:
            self.processOOT = binning.processOOT(self.binnedLC, self.binnedLC, self.binnedWav, self.opt)   
        
        self.ootSignal = self.processOOT.ootSignal     
        self.ootNoise = self.processOOT.ootNoise
             
        if self.opt.pipeline.useAllen.val ==1:
            self.ootAllen = self.processOOT.ootAllen
            self.noiseAt1hr = self.processOOT.noiseAt1hr
          
