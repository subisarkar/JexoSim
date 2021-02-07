'''
JexoSim
2.0
Recipe 3 with ability to split timeline into chunks - used to test split method

'''
from jexosim.modules import exosystem, telescope, channel, backgrounds
from jexosim.modules import detector, timeline, light_curve, systematics, noise, output 
from jexosim.pipeline.run_pipeline import pipeline_stage_1, pipeline_stage_2
from jexosim.lib.jexosim_lib import jexosim_msg, jexosim_plot, write_record
from astropy import units as u
from datetime import datetime
import pickle, os
import numpy as np


class recipe_3a(object):
    def __init__(self, opt):
        
        output_directory = opt.common.output_directory.val
        filename=""
        
        self.results_dict ={}
        self.results_dict['simulation_mode'] = opt.simulation.sim_mode.val
        self.results_dict['simulation_realisations'] = opt.simulation.sim_realisations.val
        self.results_dict['ch'] =  opt.observation.obs_channel.val 
        self.noise_dict ={}  
        self.feasibility =1

        opt.pipeline.useSignal.val=1
        opt.pipeline.use_fast.val =1
        opt.pipeline.split  = 0
        opt.noise.ApplyRandomPRNU.val=1

        opt.timeline.apply_lc.val = 0
        opt.timeline.useLDC.val = 0
        opt.pipeline.useAllen.val =1

        opt.pipeline.pipeline_auto_ap.val = 0 # for noise budget keep this to a fixed value (i.e. choose 0) so same for all sims

        opt.timeline.obs_time.val = 0*u.hr
        opt.timeline.n_exp.val = 1000.0
         
        noise_list = [0,2,3,4,5,6,7,8,9,12]
        # noise_list = [0,2,3,4,5,6,7,8,9]
        # noise_list = [2,9,12]


        start = 0 
        end = int(start + opt.no_real)      

        nb_dict = {'rn'           :[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                   'sn'           :[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                   'spat'         :[1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0],                                   
                   'spec'         :[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],                                      
                   'emm_switch'   :[1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1],                    
                   'zodi_switch'  :[1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1],    
                   'dc_switch'    :[1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                   'source_switch':[1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                   'diff'         :[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                   'jitter_switch':[1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
                   'fano'         :[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                   'noise_tag': [ 'All noise','All photon noise','Source photon noise','Dark current noise',
                        'Zodi noise','Emission noise','Read noise','Spatial jitter noise',
                        'Spectral jitter noise','Combined jitter noise','No noise - no background','No noise - all background', 'Fano noise'],  
                        'color': ['0.5','b', 'b','k','orange','pink', 'y','g','purple','r', '0.8','c', 'c']
              } 

        for i in noise_list:
            self.noise_dict[nb_dict['noise_tag'][i]] ={}
            
        for j in range(start,end):
            seed = np.random.randint(100000000)
            for i in noise_list:
    
                opt.noise.EnableReadoutNoise.val = nb_dict['rn'][i]
                opt.noise.EnableShotNoise.val = nb_dict['sn'][i]
                opt.noise.EnableSpatialJitter.val= nb_dict['spat'][i]
                opt.noise.EnableSpectralJitter.val= nb_dict['spec'][i]   
                opt.background.EnableEmission.val = nb_dict['emm_switch'][i]
                opt.background.EnableZodi.val = nb_dict['zodi_switch'][i]    
                opt.background.EnableDC.val  =  nb_dict['dc_switch'][i]
                opt.background.EnableSource.val  = nb_dict['source_switch'][i]
                opt.noise.EnableFanoNoise.val = nb_dict['fano'][i]
                opt.diff = nb_dict['diff'][i]      
                opt.noise_tag = nb_dict['noise_tag'][i]
                opt.color = nb_dict['color'][i]
                
                jexosim_msg('==========================================', 1)
                jexosim_msg('Noise source:%s'%(opt.noise_tag), 1)     
                
                np.random.seed(seed)
                opt = self.run_JexoSimA(opt)
                opt = self.run_JexoSimA1(opt)
                
                n_ndr0 = opt.n_ndr*1
                ndr_end_frame_number0 = opt.ndr_end_frame_number*1
                frames_per_ndr0 = opt.frames_per_ndr*1
                duration_per_ndr0 = opt.duration_per_ndr*1
                n_exp0 = opt.n_exp           
                            
                if n_ndr0 > 10000:
    
                    opt.pipeline.split = 1
                    if opt.diagnostics ==1 :
                       jexosim_msg ('number of NDRs > 10000: using split protocol', opt.diagnostics)
                else:
                    opt.pipeline.split = 0 

                opt.pipeline.split = 1                      

              # =============================================================================
              #  split simulation into chunks to permit computation - makes no difference to final results    
              #  =============================================================================
                if opt.pipeline.split ==1:
              
                   jexosim_msg('Splitting data series into chunks', opt.diagnostics)
                   # uses same QE grid and jitter timeline but otherwise randomoses noise
                   ndrs_per_round = opt.effective_multiaccum*int(5000/opt.multiaccum)  
                   ndrs_per_round = opt.effective_multiaccum*int(50/opt.multiaccum)  
  
                   total_chunks = len(np.arange(0, n_ndr0, ndrs_per_round))
                   
                   idx = np.arange(0, n_ndr0, ndrs_per_round) # list of starting ndrs
                   
                   for i in range(len(idx)):
                       jexosim_msg('=== Chunk %s / %s====='%(i+1, total_chunks), opt.diagnostics)
                       
                       if idx[i] == idx[-1]:
                           opt.n_ndr = n_ndr0 - idx[i]

                           opt.ndr_end_frame_number = ndr_end_frame_number0[idx[i]:]
                           opt.frames_per_ndr=  frames_per_ndr0[idx[i]:]
                           opt.duration_per_ndr = duration_per_ndr0[idx[i]:]  
                          
                       else:
                           opt.n_ndr = idx[i+1]- idx[i]

                           opt.ndr_end_frame_number = ndr_end_frame_number0[idx[i]: idx[i+1]]
                           opt.frames_per_ndr=  frames_per_ndr0[idx[i]: idx[i+1]]
                           opt.duration_per_ndr = duration_per_ndr0[idx[i]: idx[i+1]]
                  
                       opt.n_exp = int(opt.n_ndr/opt.effective_multiaccum)
                                   
                       if i == 0: 
                           opt.use_external_jitter = 0
                           
                           opt = self.run_JexoSimB(opt)
                           opt  = self.run_pipeline_stage_1(opt)  
                           if opt.pipeline.pipeline_auto_ap.val == 1:
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
                       
                       #After chunks processed, now recombine                                           
                       if i ==0:
                           data_stack = data
                           binnedLC_stack = binnedLC   
                       else:
                           data_stack = np.dstack((data_stack,data))
                           binnedLC_stack = np.vstack((binnedLC_stack, binnedLC))                  
                   
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
                       if opt.pipeline.pipeline_apply_mask.val == 1:
                           opt.pipeline.pipeline_ap_factor.val= opt.AvBest

                   binnedLC_stack  = opt.pipeline_stage_1.binnedLC
                   data_stack = opt.pipeline_stage_1.opt.data_raw
                   jexosim_plot('testvvv', opt.diagnostics,
                                ydata=binnedLC_stack.sum(axis=1) )  

            
                           
                #take binnedLC_stack and now complete through pipeline stage 2
                opt.pipeline_stage_1.binnedLC = binnedLC_stack    
                opt = self.run_pipeline_stage_2(opt)
                self.pipeline = opt.pipeline_stage_2  
                self.opt = opt
                        
                self.noise_dict[opt.noise_tag]['wl'] = self.pipeline.binnedWav
                 
                if j == start:                    
                    self.noise_dict[opt.noise_tag]['signal_std_stack'] = self.pipeline.ootNoise
                    self.noise_dict[opt.noise_tag]['signal_mean_stack'] = self.pipeline.ootSignal
                    if opt.pipeline.useAllen.val == 1:
                        self.noise_dict[opt.noise_tag]['fracNoT14_stack'] = self.pipeline.noiseAt1hr
                        
                    self.noise_dict[opt.noise_tag]['signal_std_mean'] = self.pipeline.ootNoise
                    self.noise_dict[opt.noise_tag]['signal_mean_mean'] = self.pipeline.ootSignal
                    if opt.pipeline.useAllen.val == 1:
                        self.noise_dict[opt.noise_tag]['fracNoT14_mean'] = self.pipeline.noiseAt1hr
                        
                    self.noise_dict[opt.noise_tag]['signal_std_std'] = np.zeros(len(self.pipeline.binnedWav))
                    self.noise_dict[opt.noise_tag]['signal_mean_std'] = np.zeros(len(self.pipeline.binnedWav))
                    if opt.pipeline.useAllen.val == 1:
                        self.noise_dict[opt.noise_tag]['fracNoT14_std'] = np.zeros(len(self.pipeline.binnedWav))

                    self.noise_dict[opt.noise_tag]['bad_map'] = opt.bad_map
                    self.noise_dict[opt.noise_tag]['example_exposure_image'] = opt.exp_image
                    self.noise_dict[opt.noise_tag]['pixel wavelengths'] = opt.x_wav_osr[1::3].value
                                     

                else:
                    self.noise_dict[opt.noise_tag]['signal_std_stack'] = np.vstack((self.noise_dict[opt.noise_tag]['signal_std_stack'], self.pipeline.ootNoise))
                    self.noise_dict[opt.noise_tag]['signal_mean_stack'] = np.vstack((self.noise_dict[opt.noise_tag]['signal_mean_stack'], self.pipeline.ootSignal))
                    if opt.pipeline.useAllen.val == 1:
                        self.noise_dict[opt.noise_tag]['fracNoT14_stack'] = np.vstack((self.noise_dict[opt.noise_tag]['fracNoT14_stack'], self.pipeline.noiseAt1hr))

                    self.noise_dict[opt.noise_tag]['signal_std_mean'] = self.noise_dict[opt.noise_tag]['signal_std_stack'].mean(axis=0)
                    self.noise_dict[opt.noise_tag]['signal_mean_mean'] = self.noise_dict[opt.noise_tag]['signal_mean_stack'].mean(axis=0)
                    if opt.pipeline.useAllen.val == 1:
                        self.noise_dict[opt.noise_tag]['fracNoT14_mean'] = self.noise_dict[opt.noise_tag]['fracNoT14_stack'].mean(axis=0)
                        
                    self.noise_dict[opt.noise_tag]['signal_std_std'] = self.noise_dict[opt.noise_tag]['signal_std_stack'].std(axis=0)
                    self.noise_dict[opt.noise_tag]['signal_mean_std'] = self.noise_dict[opt.noise_tag]['signal_mean_stack'].std(axis=0)
                    if opt.pipeline.useAllen.val == 1:
                        self.noise_dict[opt.noise_tag]['fracNoT14_std'] = self.noise_dict[opt.noise_tag]['fracNoT14_stack'].std(axis=0)

                self.noise_dict[opt.noise_tag]['bad_map'] = opt.bad_map
                self.noise_dict[opt.noise_tag]['example_exposure_image'] = opt.exp_image
                self.noise_dict[opt.noise_tag]['pixel wavelengths'] = opt.x_wav_osr[1::3].value
                    
                self.results_dict['noise_dic'] = self.noise_dict         
 
            # dump pickle file at end of each cycle of noise sims
            if opt.simulation.sim_output_type.val == 1:         
                time_tag = (datetime.now().strftime('%Y_%m_%d_%H%M_%S'))
                self.results_dict['time_tag'] =  time_tag
         
                # if j != start:
                #     os.remove(filename)  # delete previous temp file
                # filename = '%s/Noise_budget_%s_TEMP.pickle'%(output_directory, opt.lab)
                # with open(filename, 'wb') as handle:
                #     pickle.dump(self.results_dict , handle, protocol=pickle.HIGHEST_PROTOCOL)
                   
                if j == end-1:
                    # os.remove(filename)  # delete previous temp file
                    filename = '%s/Noise_budget_%s_%s.pickle'%(output_directory, opt.lab, time_tag)
                    with open(filename, 'wb') as handle:
                        pickle.dump(self.results_dict , handle, protocol=pickle.HIGHEST_PROTOCOL)
     
                        jexosim_msg('Results in %s'%(filename), 1)
                        self.filename = 'Noise_budget_%s_%s.pickle'%(opt.lab, time_tag)

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
    
