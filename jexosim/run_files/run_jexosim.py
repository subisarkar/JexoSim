'''
JexoSim 
2.0
Run file
v1.0

'''
from jexosim.classes.options import Options
from jexosim.classes.params import Params
from jexosim.run_files.recipe_1 import recipe_1
from jexosim.run_files.recipe_1a import recipe_1a
from jexosim.run_files.recipe_2 import recipe_2
from jexosim.run_files.recipe_2b import recipe_2b
from jexosim.run_files.recipe_3 import recipe_3
from jexosim.run_files.recipe_3a import recipe_3a
from jexosim.run_files.recipe_4 import recipe_4

from jexosim.run_files.recipe_2_no_pipeline import recipe_2_no_pipeline
from jexosim.run_files.recipe_1_no_pipeline import recipe_1_no_pipeline
from jexosim.run_files.recipe_1_intermediate import recipe_1_intermediate
from jexosim.run_files.recipe_2_intermediate import recipe_2_intermediate

from jexosim.run_files import results
from jexosim.generate.gen_planet_xml_file import make_planet_xml_file
from jexosim.lib.jexosim_lib import jexosim_msg
import jexosim
import os, sys
import time

 
#====Load defauls from XML files and load input text file with user-defined adjustments==========================================================================
# 
def run(params_file):
    
    aa = time.time()   
# for a in [0]:
    # # params_file = 'jexosim_input_params_ex7 copy.txt'
    # params_file = 'jexosim_input_params_ex3.txt'
   
    jexosim_msg('JexoSim is running!\n', 1)    
    jexosim_msg('User-defined input parameter file: %s\n  '%(params_file), 1) 
    jexosim_path =  os.path.dirname((os.path.dirname(jexosim.__file__)))
    jexosim_msg('Reading default common configuration file ... \n', 1)
    common_file = 'jexosim/xml_files/JWST.xml'
      
    opt = Options(filename='%s/%s'%(jexosim_path,common_file)).opt 
    opt.common_file = common_file
      
    jexosim_msg('Apply user-defined adjustments to default... \n', 1)
    
    paths_file = '%s/jexosim/input_files/jexosim_paths.txt'%(jexosim_path)
    params_file = '%s/jexosim/input_files/%s'%(jexosim_path, params_file)
    
    #read in path information set by user
    params_to_opt = Params(opt, paths_file, 0)  
    paths = params_to_opt.params
    opt = params_to_opt.opt # adjust default values to user defined ones
 
    # read in input parameters for this simulation
    params_to_opt = Params(opt, params_file, 1)  
    input_params = params_to_opt.params
    opt = params_to_opt.opt # adjust default values to user defined ones   
          
    jexosim_msg('Instrument mode %s \n'%(opt.observation.obs_channel.val), 1)    
    jexosim_msg('Reading default instrument mode configuration file \n', 1)
    idx =  opt.observation.obs_channel.val.find('_')
    ch_folder = opt.observation.obs_channel.val[:idx]
    mode = opt.observation.obs_channel.val[idx+1:]

    ch_file = '%s/xml_files/%s/%s.xml'%(opt.__path__, ch_folder, mode)
    
    opt2 = Options(ch_file).opt
    for key in opt2.__dict__:
          setattr(opt, str(key), opt2.__dict__[key])
    opt.ICF = ["%s/%s"%(opt.__path__,common_file), ch_file] 
   
    jexosim_msg('Applying user-defined adjustments to default values... \n', 1)      
    params_to_opt = Params(opt, params_file, 2)
    input_params = params_to_opt.params
    opt = params_to_opt.opt # adjust defaulf values to user defined ones 
    
    pl = opt.exosystem_params.planet_name.val
    # an XML file is made for the planet (if one exists code will use that unless planet_file_renew =1 )
    make_planet_xml_file(opt, pl)    
    jexosim_msg('Reading exosystem parameters file ... \n', 1)
    # overrides defaults in JWST.xml file
    ex_file = '%s/xml_files/exosystems/%s.xml'%(opt.__path__, pl)
    opt3 = Options(ex_file).opt
    for key in opt3.__dict__:
        setattr(opt, str(key), opt3.__dict__[key])

    if opt.exosystem_params.planet_use_database.val == 0:
        pl =  input_params['user_defined_planet_name']
        
 
    #==============================================================================
    # Set noise source from noise budget matrix - use one noise source only as not in a loop
    #==============================================================================      
    i = int(opt.noise.sim_noise_source.val) # noise option - choose 0 for all noise - default      
    nb_dict = {'rn'                 :[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
               'sn'                 :[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
               'spat'               :[1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],                                   
               'spec'               :[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],                                      
               'emm_switch'         :[1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0],                    
               'zodi_switch'        :[1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0],    
               'dc_switch'          :[1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
               'source_switch'      :[1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
               'diff'               :[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1],
               'jitter_switch'      :[1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
               'fano'               :[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
               'sunshield_switch'   :[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
               'noise_tag': [ 'All noise','All photon noise','Source photon noise','Dark current noise',
                        'Zodi noise','Emission noise','Read noise','Spatial jitter noise',
                        'Spectral jitter noise','Combined jitter noise','No noise - no background','No noise - all background', 'Fano noise', 'Sunshield noise'],  
                        'color': ['0.5','b', 'b','k','orange','pink', 'y','g','purple','r', '0.8','c', 'c', 'brown']
              } 
                

    opt.noise.EnableReadoutNoise.val = nb_dict['rn'][i]
    opt.noise.EnableShotNoise.val = nb_dict['sn'][i]
    opt.noise.EnableSpatialJitter.val= nb_dict['spat'][i]
    opt.noise.EnableSpectralJitter.val= nb_dict['spec'][i]
    opt.noise.EnableFanoNoise.val= nb_dict['fano'][i]
    opt.background.EnableEmission.val = nb_dict['emm_switch'][i]
    opt.background.EnableZodi.val = nb_dict['zodi_switch'][i]  
    opt.background.EnableSunshield.val = nb_dict['sunshield_switch'][i]   
    opt.background.EnableDC.val  =  nb_dict['dc_switch'][i]
    opt.background.EnableSource.val  = nb_dict['source_switch'][i]
    opt.diff = nb_dict['diff'][i]      
    opt.noise_tag = nb_dict['noise_tag'][i]
    opt.color = nb_dict['color'][i]
    
           
    #==============================================================================
    #  Run with right recipe
    #==============================================================================
    opt.lab = '%s_%s'%(opt.observation.obs_inst_config.val, pl)
    
    opt.lab = opt.lab.replace(' + ','_')
    opt.lab = opt.lab.replace(' ','_')
    
    opt.no_real = opt.simulation.sim_realisations.val
    opt.diagnostics = opt.simulation.sim_diagnostics.val
    opt.input_params = input_params
    opt.jexosim_path = jexosim_path
    opt.params_file_path = params_file 
    if opt.noise.sim_prnu_rms.val !=0:
        opt.noise.ApplyPRNU.val = 1
    else:
        opt.noise.ApplyPRNU.val = 0
    
    jexosim_msg(('Simulation mode %s'%(int(opt.simulation.sim_mode.val))), 1)    
    jexosim_msg(f'{opt.lab}', 1)
   
    if opt.simulation.sim_mode.val == 1:
        if opt.simulation.sim_output_type.val == 1:
            recipe  = recipe_1(opt) # replace with 1a to test split method
        elif opt.simulation.sim_output_type.val == 2:   
            recipe  = recipe_1_no_pipeline(opt)  
        elif opt.simulation.sim_output_type.val == 3:  
            recipe  = recipe_1_intermediate(opt)  
            
    if opt.simulation.sim_mode.val == 2:
            if opt.simulation.sim_output_type.val == 1:
                recipe  = recipe_2(opt)
            elif opt.simulation.sim_output_type.val == 2:
                recipe  = recipe_2_no_pipeline(opt)  
            if opt.simulation.sim_output_type.val == 3: #
                recipe  = recipe_2_intermediate(opt)   
                
    if opt.simulation.sim_mode.val == 3: #noise budget
            if opt.simulation.sim_output_type.val == 1:
                recipe  = recipe_3(opt) # replace with 3a to test split method
            elif opt.simulation.sim_output_type.val !=1:
                jexosim_msg('This output option is not currently available for noise budgets',1)
                sys.exit()
  
    if opt.simulation.sim_mode.val == 4: #same as 1 but no Allen analysis
        if opt.simulation.sim_output_type.val == 1:
            recipe  = recipe_4(opt) 
        elif opt.simulation.sim_output_type.val == 2:   
            recipe  = recipe_1_no_pipeline(opt) 
        elif opt.simulation.sim_output_type.val == 3:  
            recipe  = recipe_1_intermediate(opt)          
          
    print ('time to complete', time.time() - aa)     
          
    if recipe.feasibility ==1:    
        if opt.simulation.sim_output_type.val == 1:
            results_file = recipe.filename
            results.run(results_file)
    else:
        jexosim_msg('No results', 1)
              
    #==============================================================================
    #      Store results
    #==============================================================================
   
if __name__ == "__main__":      
  

    # run('jexosim_input_params_ex5 copy.txt')  

    run('jexosim_input_params_ex_miri_test.txt')
    # run('jexosim_input_params_ex_nirspec_test.txt')
    # run('jexosim_input_params_ex_nircam_test.txt')
    # run('jexosim_input_params_ex_niriss_test.txt')
 