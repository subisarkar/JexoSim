'''
JexoSim
2.0
Recipe 2 :
Monte Carlo full transit simulation returning transit depths and noise on transit dept per spectral bin

Each realisation run the stage 2 JexoSim routine with a new noise randomization
The if the pipeline_auto_ap=1, the aperture mask size may vary with each realisation
This should not matter since final measurement is the fractional transit depth

'''

import numpy as np
import time, os, pickle
from datetime import datetime
from jexosim.modules import exosystem, telescope, channel, backgrounds
from jexosim.modules import detector, timeline, light_curve, systematics, noise
from jexosim.pipeline.run_pipeline import pipeline_stage_1, pipeline_stage_2
from jexosim.lib.jexosim_lib import jexosim_msg, jexosim_plot, write_record
import gc


class recipe_2(object):
    def __init__(self, opt):
        
        output_directory = opt.common.output_directory.val
        filename=""
        runtag = int(np.random.uniform(0,100000))
        
        
        self.results_dict ={}
        self.results_dict['simulation_mode'] = opt.simulation.sim_mode.val
        self.results_dict['simulation_realisations'] = opt.simulation.sim_realisations.val
        self.results_dict['ch'] =  opt.observation.obs_channel.val 

        opt.pipeline.useSignal.val=0
        opt.simulation.sim_use_fast.val =1
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
 
        # np.save('/Users/user1/Desktop/fp_signal.npy', opt.fp_signal[1::3,1::3].value)
        # np.save('/Users/user1/Desktop/fp_wav.npy', opt.x_wav_osr[1::3].value)
                    
                    
        # xxxx
        
                                      
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
               
               print (opt.diagnostics)
               
               
               
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
                       jexosim_msg('=== Realisation %s Chunk %s / %s====='%(j, i+1, total_chunks), 1)
                       jexosim_msg (opt.lab, 1) 
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
                           data_stack = data*1
                           binnedLC_stack = binnedLC*1    
                       else:
                           data_stack = np.dstack((data_stack,data))
                           binnedLC_stack = np.vstack((binnedLC_stack,binnedLC))  
                           
                       del data
                       del binnedLC

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
                   
                
               opt.pipeline_stage_1.binnedLC = binnedLC_stack     
               opt = self.run_pipeline_stage_2(opt)
               pipeline = opt.pipeline_stage_2
       
               p = pipeline.transitDepths
               if j == start:
                   p_stack = p
               else:
                   p_stack = np.vstack((p_stack,p))
                                      
               jexosim_msg ("time to complete realization %s %s"%(j, time.time()-pp ) ,opt.diagnostics)
        

               self.results_dict['wl'] = pipeline.binnedWav   
               self.results_dict['input_spec'] = opt.cr
               self.results_dict['input_spec_wl'] = opt.cr_wl
              
               if j==start:  # if only one realisation slightly different format
                   self.results_dict['p_stack'] = np.array(p)
                   self.results_dict['p_std']= np.zeros(len(p))
                   self.results_dict['p_mean'] = np.array(p)             
               else:
                   self.results_dict['p_stack'] = np.vstack((self.results_dict['p_stack'], p))
                   self.results_dict['p_std'] = self.results_dict['p_stack'].std(axis=0)  
                   self.results_dict['p_mean'] = self.results_dict['p_stack'].mean(axis=0)

                      
               time_tag = (datetime.now().strftime('%Y_%m_%d_%H%M_%S'))
                   
               self.results_dict['time_tag'] =  time_tag
               self.results_dict['bad_map'] = opt.bad_map
               self.results_dict['example_exposure_image'] = opt.exp_image
               self.results_dict['pixel wavelengths'] = opt.x_wav_osr[1::3].value
               self.results_dict['focal_plane_star_signal'] = opt.fp_signal[1::3,1::3].value
                
            
         
               # fq = '/Users/user1/Desktop/tempGit/JexoSim_A/output/Full_eclipse_MIRI_LRS_slitless_SLITLESSPRISM_FAST_HD_209458_b_2021_07_18_0852_28.pickle'


               # with open(fq, 'rb') as handle:
               #   rd = pickle.load(handle)             
               # rd['focal_plane_star_signal'] = opt.fp_signal[1::3,1::3]
               # rd['pixel wavelengths'] = opt.x_wav_osr[1::3].value
               # with open(fq, 'wb') as handle:
               #         pickle.dump(rd , handle, protocol=pickle.HIGHEST_PROTOCOL)   
    
    # run('Full_transit_NIRSpec_BOTS_G140M_F100LP_SUB2048_NRSRAPID_K2-18_b_2021_07_18_1402_38.pickle',0)
    # run('Full_transit_NIRSpec_BOTS_G235M_F170LP_SUB2048_NRSRAPID_K2-18_b_2021_07_18_0709_24.pickle', 2)
    # run('Full_transit_NIRSpec_BOTS_G395M_F290LP_SUB2048_NRSRAPID_K2-18_b_2021_07_17_2339_51.pickle',4)
    
    # run('Full_eclipse_NIRCam_TSGRISM_F444W_SUBGRISM64_4_output_RAPID_HD_209458_b_2021_07_19_0654_29.pickle',0)
    # run('Full_eclipse_NIRCam_TSGRISM_F322W2_SUBGRISM64_4_output_RAPID_HD_209458_b_2021_07_19_0359_08.pickle',1)
    # run('Full_eclipse_MIRI_LRS_slitless_SLITLESSPRISM_FAST_HD_209458_b_2021_07_18_0852_28.pickle', 2)
 
     
         
                       
    
               if j != start:
                    os.remove(filename)  # delete previous temp file
     
               filename = '%s/Full_transit_%s_TEMP%s.pickle'%(output_directory, opt.lab, runtag)
               if opt.observation.obs_type.val == 2:
                    filename = '%s/Full_eclipse_%s_TEMP%s.pickle'%(output_directory, opt.lab, runtag)
                       
               with open(filename, 'wb') as handle:
                    pickle.dump(self.results_dict , handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
               del pipeline
               del binnedLC_stack
               gc.collect()
                   
                
           os.remove(filename)  # delete previous temp file
            # write final file
           filename = '%s/Full_transit_%s_%s.pickle'%(output_directory, opt.lab, time_tag)
           if opt.observation.obs_type.val == 2:
               filename = '%s/Full_eclipse_%s_%s.pickle'%(output_directory, opt.lab, time_tag)
               
           with open(filename, 'wb') as handle:
                  pickle.dump(self.results_dict , handle, protocol=pickle.HIGHEST_PROTOCOL)
        
           jexosim_msg('Results in %s'%(filename), 1)
           self.filename = 'Full_transit_%s_%s.pickle'%(opt.lab, time_tag)
           if opt.observation.obs_type.val == 2:
               self.filename = 'Full_eclipse_%s_%s.pickle'%(opt.lab, time_tag)
               
           write_record(opt, output_directory, self.filename, opt.params_file_path)
            
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
             
    def run_pipeline_stage_2(self, opt):    
      jexosim_msg('Pipeline stage 2', 1)
      opt.pipeline_stage_2 = pipeline_stage_2(opt)             
      return opt 
    
