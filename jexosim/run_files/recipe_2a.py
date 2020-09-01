'''
JexoSim
2.0
Recipe 2 :
Monte Carlo full transit simulation returning transit depths and noise on transit dept per spectral bin
Result is a FITS file only
'''

import numpy as np
import time, os
from datetime import datetime
from jexosim.modules import exosystem, telescope, channel, backgrounds, output
from jexosim.modules import detector, timeline, light_curve, systematics, noise
from jexosim.lib.jexosim_lib import jexosim_msg, jexosim_plot, write_record


class recipe_2a(object):
    def __init__(self, opt):
        
        output_directory = opt.common.output_directory.val
        
        opt.pipeline.useSignal.val=0
        opt.pipeline.use_fast.val =0  # this is important to return the right size of image
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
           lc0 = opt.lc_original*1
           ndr_end_frame_number0 = opt.ndr_end_frame_number*1
           frames_per_ndr0 = opt.frames_per_ndr*1
           duration_per_ndr0 = opt.duration_per_ndr*1
           
           if n_ndr0 > 20000:
               opt.pipeline.split = 1
               if opt.diagnostics ==1 :
                   jexosim_msg ('number of NDRs > 20000: using split protocol', opt.diagnostics)
           else:
               opt.pipeline.split = 0 
       
           for j in range(start, end):
               
               if (opt.no_real-start) > 1:
                   jexosim_msg ("", 1)    
                   jexosim_msg ("============= REALIZATION %s ============="%(j), 1)
                   jexosim_msg (opt.lab, 1)                   
                   jexosim_msg ("", 1) 
               
               pp = time.time()
               # split simulation into chunks to permit computation - makes no difference to final results    
               if opt.pipeline.split ==1:
                   # uses same QE grid and jitter timeline but otherwise randomoses noise
                   ndrs_per_round = opt.effective_multiaccum*int(1000/opt.multiaccum)      
                   idx_list =[]
                   for i in range(0, n_ndr0, ndrs_per_round):
                       idx_list.append(i)        
    
                   for i in range(len(idx_list)):
                       if idx_list[i] == idx_list[-1]:
                           opt.n_ndr = n_ndr0 - idx_list[i]
                           opt.lc_original = lc0[:,idx_list[i]:]
                           opt.ndr_end_frame_number = ndr_end_frame_number0[idx_list[i]:]
                           opt.frames_per_ndr=  frames_per_ndr0[idx_list[i]:]
                           opt.duration_per_ndr = duration_per_ndr0[idx_list[i]:]                           
                       else:
                           opt.n_ndr = idx_list[i+1]- idx_list[i]
                           opt.lc_original = lc0[:,idx_list[i]:idx_list[i+1]]
                           opt.ndr_end_frame_number = ndr_end_frame_number0[idx_list[i]: idx_list[i+1]]
                           opt.frames_per_ndr=  frames_per_ndr0[idx_list[i]: idx_list[i+1]]
                           opt.duration_per_ndr = duration_per_ndr0[idx_list[i]: idx_list[i+1]]
                           
                       opt.n_exp = int(opt.n_ndr/opt.multiaccum)
                                   
                       if i == 0:             
                           opt.use_external_jitter = 0
                       else:                     
                           opt.use_external_jitter = 1 # uses the jitter timeline from the first realization
                           if (opt.noise.EnableSpatialJitter.val  ==1 or opt.noise.EnableSpectralJitter.val  ==1 or opt.noise.EnableAll.val == 1) and opt.noise.DisableAll.val != 1:
                               opt.input_yaw_jitter, opt.input_pitch_jitter, opt._input_frame_osf = opt.yaw_jitter, opt.pitch_jitter, opt.frame_osf 
                 
                       opt = self.run_JexoSimB(opt)
                       
                       data = opt.data
                                                                  
                       if i ==0:
                           data_stack = data
                     
                       else:
                           data_stack = np.dstack((data_stack,data))

     
                   opt.n_ndr = n_ndr0
                   opt.lc_original = lc0
                   opt.ndr_end_frame_number =  ndr_end_frame_number0
                   opt.frames_per_ndr  = frames_per_ndr0
                   opt.duration_per_ndr = duration_per_ndr0
                   opt.n_exp = int(opt.n_ndr/opt.multiaccum).value
                                                            
               elif opt.pipeline.split ==0:
   
                   opt  = self.run_JexoSimB(opt)                  
                   data_stack = opt.data
               
               opt.data = data_stack
               
               filename = output.run(opt)
                      
               write_record(opt, output_directory, filename, opt.params_file_path)               
      
            
    def run_JexoSimA(self, opt):
      exosystem.run(opt) 
      telescope.run(opt) 
      channel.run(opt)  
      backgrounds.run(opt)     
      detector.run(opt)
      if opt.observation_feasibility ==1: # if detector does not saturate continue
          timeline.run(opt)    
          light_curve.run(opt)     
          return opt       
      else: # if detector saturates end sim      
          return opt 
      
    def run_JexoSimB(self, opt):
      systematics.run(opt)   
      noise.run(opt)                 
      return opt
           
    def run_pipeline_stage_1(self, opt):
      opt.pipeline_stage_1 = pipeline_stage_1(opt)   
      return opt  
             
    def run_pipeline_stage_2(self, opt):        
      opt.pipeline_stage_2 = pipeline_stage_2(opt)             
      return opt 
    