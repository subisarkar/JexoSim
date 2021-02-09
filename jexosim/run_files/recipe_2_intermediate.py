'''
JexoSim
2.0
Recipe 2: Full transit simulation
Pipeline Stage 1 processing only, with binned light curves returned in FITS format

'''

import numpy as np
import time, os, pickle
from datetime import datetime
from jexosim.modules import exosystem, telescope, channel, backgrounds, output
from jexosim.modules import detector, timeline, light_curve, systematics, noise
from jexosim.pipeline.run_pipeline import pipeline_stage_1
from jexosim.lib.jexosim_lib import jexosim_msg, jexosim_plot, write_record
from astropy import units as u


class recipe_2_intermediate(object):
    def __init__(self, opt):
        
        output_directory = opt.common.output_directory.val
        filename=""
        
        self.results_dict ={}
        self.results_dict['simulation_mode'] = opt.simulation.sim_mode.val
        self.results_dict['simulation_realisations'] = opt.simulation.sim_realisations.val
        self.results_dict['ch'] =  opt.observation.obs_channel.val 

        opt.pipeline.useSignal.val=0
        opt.pipeline.use_fast.val =1
        opt.pipeline.split  = 0
        opt.noise.ApplyRandomPRNU.val=1

        opt.timeline.apply_lc.val = 1
        opt.timeline.useLDC.val = 1
        opt.pipeline.useAllen.val =0
        opt.pipeline.fit_gamma.val  =0 #keep zero for uncert on p
    
        start = 0 
        end = int(start + opt.no_real)
        
        if (opt.no_real-start) > 1:
            jexosim_msg ("Monte Carlo selected", 1) 
                           
        opt = self.run_JexoSimA(opt)
                                      
        if opt.observation_feasibility ==0:      
           jexosim_msg ("Observation not feasible...", opt.diagnostics) 
           self.feasibility = 0
        else:
           self.feasibility = 1
           n_ndr0 = opt.n_ndr*1
           
           ndr_end_frame_number0 = opt.ndr_end_frame_number*1
           frames_per_ndr0 = opt.frames_per_ndr*1
           duration_per_ndr0 = opt.duration_per_ndr*1
           n_exp0 = opt.n_exp
           lc0 = opt.lc_original*1   # important this happens at this stage      
                        
           if n_ndr0 > 10000:

               opt.pipeline.split = 1
               if opt.diagnostics ==1 :
                   jexosim_msg ('number of NDRs > 10000: using split protocol', opt.diagnostics)
           else:
               opt.pipeline.split = 0 
               
           # #delete    
           # opt.pipeline.split = 1 
           
           for j in range(start, end):
               
               if (opt.no_real-start) > 1:
                   jexosim_msg ("", 1)    
                   jexosim_msg ("============= REALIZATION %s ============="%(j), 1)
                   jexosim_msg (opt.lab, 1)                   
                   jexosim_msg ("", 1) 
               
               pp = time.time()
               
               opt = self.run_JexoSimA1(opt)  # set QE grid for this realization
               jexosim_msg ("QE variations set", 1) 
               jexosim_msg ("Number of exposures %s"%(n_exp0), 1) 
                        
# =============================================================================
#  # split simulation into chunks to permit computation - makes no difference to final results    
# =============================================================================
               if opt.pipeline.split ==1:
              
                   jexosim_msg('Splitting data series into chunks', opt.diagnostics)
                   # uses same QE grid and jitter timeline but otherwise randomoses noise
                   ndrs_per_round = opt.effective_multiaccum*int(5000/opt.multiaccum)  
                   # ndrs_per_round = opt.effective_multiaccum*int(500/opt.multiaccum)  
  
                   total_chunks = len(np.arange(0, n_ndr0, ndrs_per_round))
                   
                   idx = np.arange(0, n_ndr0, ndrs_per_round) # list of starting ndrs
                   
                   for i in range(len(idx)):
                       jexosim_msg('=== Realisation %s Chunk %s / %s====='%(j, i+1, total_chunks), opt.diagnostics)
                       
                       if idx[i] == idx[-1]:
                           opt.n_ndr = n_ndr0 - idx[i]
                           opt.lc_original = lc0[:,idx[i]:]
                           opt.ndr_end_frame_number = ndr_end_frame_number0[idx[i]:]
                           opt.frames_per_ndr=  frames_per_ndr0[idx[i]:]
                           opt.duration_per_ndr = duration_per_ndr0[idx[i]:]  
                          
                       else:
                           opt.n_ndr = idx[i+1]- idx[i]
                           opt.lc_original = lc0[:,idx[i]:idx[i+1]]
                           print ('idx start......', idx[i])
                           
                           opt.ndr_end_frame_number = ndr_end_frame_number0[idx[i]: idx[i+1]]
                           opt.frames_per_ndr=  frames_per_ndr0[idx[i]: idx[i+1]]
                           opt.duration_per_ndr = duration_per_ndr0[idx[i]: idx[i+1]]
                  
                       opt.n_exp = int(opt.n_ndr/opt.effective_multiaccum)
                           
                       if i == 0:
                           opt.pipeline.pipeline_auto_ap.val= 1
                           opt.use_external_jitter = 0
                           
                           opt = self.run_JexoSimB(opt)
                           opt  = self.run_pipeline_stage_1(opt)  
                           
                           opt.pipeline.pipeline_ap_factor.val= opt.AvBest 
                           if (opt.noise.EnableSpatialJitter.val  ==1 or opt.noise.EnableSpectralJitter.val  ==1 or opt.noise.EnableAll.val == 1) and opt.noise.DisableAll.val != 1:
                               opt.input_yaw_jitter, opt.input_pitch_jitter, opt._input_frame_osf = opt.yaw_jitter, opt.pitch_jitter, opt.frame_osf                     
                                                      
                       else:
                           opt.pipeline.pipeline_auto_ap.val = 0
                           opt.use_external_jitter = 1 # uses the jitter timeline from the first realization

                           opt = self.run_JexoSimB(opt)
                           opt  = self.run_pipeline_stage_1(opt)
                       
                                  
                       jexosim_msg('Aperture used %s'%(opt.pipeline.pipeline_ap_factor.val), opt.diagnostics)
                       binnedLC = opt.pipeline_stage_1.binnedLC
                       data = opt.pipeline_stage_1.opt.data_raw
                                                                  
                       if i ==0:
                           data_stack = data
                           binnedLC_stack = binnedLC    
                       else:
                           data_stack = np.dstack((data_stack,data))
                           binnedLC_stack = np.vstack((binnedLC_stack,binnedLC))                  

                   aa = data_stack.sum(axis=0)
                   bb = aa.sum(axis=0)
                   jexosim_plot('test_from_sim', opt.diagnostics,
                            ydata=bb[opt.effective_multiaccum::opt.effective_multiaccum] )
                   aa = binnedLC_stack.sum(axis=1)
                   jexosim_plot('test_from_pipeline', opt.diagnostics,
                                        ydata=aa)                            

                   opt.n_ndr  = n_ndr0             
                   opt.ndr_end_frame_number  = ndr_end_frame_number0  
                   opt.frames_per_ndr  = frames_per_ndr0 
                   opt.duration_per_ndr = duration_per_ndr0
                   opt.n_exp = n_exp0                                
                           
               elif opt.pipeline.split ==0:

                   opt  = self.run_JexoSimB(opt)
                   if j==start:  # first realization sets the ap, then the other use the same one
                       opt.pipeline.pipeline_auto_ap.val= 1
                   else:
                       opt.pipeline.pipeline_auto_ap.val= 0
                   opt  = self.run_pipeline_stage_1(opt)
                   if j==start:  # first realization sets the ap, then the other use the same one
                       opt.pipeline.pipeline_ap_factor.val= opt.AvBest

        
                   binnedLC_stack  = opt.pipeline_stage_1.binnedLC                   
                   jexosim_plot('testvvv', opt.diagnostics,
                                ydata=binnedLC_stack.sum(axis=1) )
                   
# =============================================================================
# Now put intermediate data (binned light curves) into FITS files
# =============================================================================
               opt.pipeline_stage_1.binnedLC*=u.electron
               opt.pipeline_stage_1.binnedWav*=u.um
  
               self.filename = output.run(opt)
                
               write_record(opt, output_directory, self.filename, opt.params_file_path)

               jexosim_msg('File saved as %s/%s'%(output_directory, self.filename), 1)
                
                             
    def run_JexoSimA(self, opt):
      jexosim_msg('Exosystem', 1)
      exosystem.run(opt)
      jexosim_msg('Telescope', 1)
      telescope.run(opt)
      jexosim_msg('Channel', 1)
      channel.run(opt)
      jexosim_msg('Backgrounds', 1)
      backgrounds.run(opt) 
      jexosim_msg('Detector', 1)
      detector.run(opt)
      if opt.observation_feasibility ==1: # if detector does not saturate continue
          jexosim_msg('Timeline', 1)
          timeline.run(opt)
          jexosim_msg('Light curve', 1)
          light_curve.run(opt)     
          return opt       
      else: # if detector saturates end sim      
          return opt

    def run_JexoSimA1(self, opt):
      jexosim_msg('Systematics', 1)
      systematics.run(opt)               
      return opt
        
    def run_JexoSimB(self, opt):
      jexosim_msg('Noise', 1)
      noise.run(opt)                 
      return opt
           
    def run_pipeline_stage_1(self, opt):
      jexosim_msg('Pipeline stage 1', 1)
      opt.pipeline_stage_1 = pipeline_stage_1(opt)   
      return opt  
             
