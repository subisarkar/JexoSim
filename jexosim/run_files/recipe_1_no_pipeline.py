'''
jexosim
2.0
Recipe 1 :
OOT simulation

This recipe only generates raw data to be used in a subsequent pipeline.

'''

import numpy as np
import time, os, pickle
from datetime import datetime
from jexosim.modules import exosystem, telescope, channel, backgrounds, output
from jexosim.modules import detector, timeline, light_curve, systematics, noise
from jexosim.lib.jexosim_lib import jexosim_msg, jexosim_plot
from jexosim.lib.output_lib import write_record_no_pipeline
from astropy import units as u

class recipe_1_no_pipeline(object):
    def __init__(self, opt):
        
        output_directory = opt.common.output_directory.val
        filename =""
        
        self.results_dict ={}
        self.results_dict['simulation_mode'] = opt.simulation.sim_mode.val
        self.results_dict['simulation_realisations'] = opt.simulation.sim_realisations.val
        self.results_dict['ch'] =  opt.observation.obs_channel.val 
        self.noise_dict ={}    
   
        opt.pipeline.useSignal.val=0
        opt.pipeline.split  = 0
        opt.noise.ApplyRandomPRNU.val=1
                      
        opt.timeline.apply_lc.val = 0
        opt.timeline.useLDC.val = 0
        opt.pipeline.useAllen.val =0
        opt.timeline.use_T14.val=0
        opt.timeline.obs_time.val = 0*u.hr # 0 means n_exp overides obs_time
        opt.timeline.n_exp.val = 1000 # uses 1000 exposures 
        # opt.timeline.n_exp.val = 50
     
        noise_type = int(opt.noise.sim_noise_source.val)
   
               
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
                   'noise_tag': [ 'All noise','All photon noise','Source photon noise','Dark current noise',
                        'Zodi noise','Emission noise','Read noise','Spatial jitter noise',
                        'Spectral jitter noise','Combined jitter noise','No noise - no background','No noise - all background', 'Fano noise'],  
                        'color': ['0.5','b', 'b','k','orange','pink', 'y','g','purple','r', '0.8', 'c']
              } 
               
        opt.noise.EnableReadoutNoise.val = nb_dict['rn'][noise_type]
        opt.noise.EnableShotNoise.val = nb_dict['sn'][noise_type]
        opt.noise.EnableSpatialJitter.val= nb_dict['spat'][noise_type]
        opt.noise.EnableSpectralJitter.val= nb_dict['spec'][noise_type] 

        opt.background.EnableEmission.val = nb_dict['emm_switch'][noise_type]
        opt.background.EnableZodi.val = nb_dict['zodi_switch'][noise_type]    
        opt.background.EnableDC.val  =  nb_dict['dc_switch'][noise_type]
        opt.background.EnableSource.val  = nb_dict['source_switch'][noise_type]
        opt.diff = nb_dict['diff'][noise_type]      
        opt.noise_tag = nb_dict['noise_tag'][noise_type]
        opt.color = nb_dict['color'][noise_type]
                        
        self.noise_dict[nb_dict['noise_tag'][noise_type]] ={}
        
        jexosim_msg ("Noise type: %s"%(nb_dict['noise_tag'][noise_type]), 1) 
  
        start = 0 
        end = int(start + opt.no_real)

        for j in range(start, end):
                        
            if (opt.no_real-start) > 1:
                   jexosim_msg ("", 1)    
                   jexosim_msg ("============= REALIZATION %s ============="%(j), 1)
                   jexosim_msg (opt.lab, 1)                   
                   jexosim_msg ("", 1)  
            
            opt = self.run_jexosimA(opt)
            opt = self.run_jexosimA1(opt)
            jexosim_msg ("QE variations set", 1) 
            jexosim_msg ("Number of exposures %s"%(opt.n_exp), 1) 

            if opt.observation_feasibility ==0:      
                jexosim_msg ("Observation not feasible...", opt.diagnostics) 
                self.feasibility = 0
            else:
                self.feasibility = 1
                              
                opt = self.run_jexosimB(opt)
            
            self.filename = output.run(opt)
        
            write_record_no_pipeline(opt, output_directory, self.filename, opt.params_file_path)
            jexosim_msg('File saved as %s/%s'%(output_directory, self.filename), 1)

         
    def run_jexosimA(self, opt):
      jexosim_msg('exosystem', 1)
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

    def run_jexosimA1(self, opt):
      jexosim_msg('Systematics', 1)
      systematics.run(opt)               
      return opt
        
    def run_jexosimB(self, opt):
      jexosim_msg('Noise', 1)
      noise.run(opt)                 
      return opt
           

