'''
JexoSim 
2.0
Run file
v1.0

'''
from jexosim.classes.options import Options
from jexosim.classes.params import Params
from jexosim.run_files.recipe_1 import recipe_1
from jexosim.run_files.recipe_2 import recipe_2
from jexosim.run_files.recipe_2a import recipe_2a
from jexosim.run_files.recipe_3 import recipe_3
from jexosim.run_files import results
from jexosim.activate.gen_planet_xml_file import make_planet_xml_file
from jexosim.lib.jexosim_lib import jexosim_msg
import jexosim
import os

 
#====Load defauls from XML files and load input text file with user-defined adjustments==========================================================================
# 
def run(params_file):
    
# for a in [0]:
    # params_file = 'jexosim_input_params_ex2.txt'
    
    jexosim_msg('JexoSim is running!\n', 1)    
    jexosim_msg('User-defined input parameter file: %s\n '%(params_file), 1) 
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
    ch = opt.observation.obs_channel.val
    ch_file = '%s/xml_files/%s.xml'%(opt.__path__, ch)
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
 
    jexosim_msg("Planet selected: %s... \n"%(pl), 1)  
    jexosim_msg('Reading exosystem parameters file ... \n', 1)
    # overrides defaults in JWST.xml file
    ex_file = '%s/xml_files/exosystems/%s.xml'%(opt.__path__, pl)
    
    opt3 = Options(ex_file).opt
    for key in opt3.__dict__:
        setattr(opt, str(key), opt3.__dict__[key])
 
    #==============================================================================
    # Set noise source from noise budget matrix - use one noise source only as not in a loop
    #==============================================================================      
    i = int(opt.noise.sim_noise_source.val) # noise option - choose 0 for all noise - default      
    nb_dict = {'rn'           :[1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           'sn'           :[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
           'spat'         :[1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],                                   
           'spec'         :[1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],                                      
           'emm_switch'   :[1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],                           
           'zodi_switch'  :[1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],       
           'dc_switch'    :[1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
           'source_switch':[1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1],
           'diff'         :[0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
           'jitter_switch':[1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
           'noise_tag': [ 'All noise','All photon noise','Source photon noise','Dark current noise',
                    'Zodi noise','Emission noise','Read noise','Spatial jitter noise',
                    'Spectral jitter noise','Combined jitter noise','No noise - no background','No noise - all background'],  
                    'color': ['0.5','b', 'b','k','orange','pink', 'y','g','purple','r', '0.8','c']
          }  
                            
    opt.noise.EnableReadoutNoise.val = nb_dict['rn'][i]
    opt.noise.EnableShotNoise.val = nb_dict['sn'][i]
    opt.noise.EnableSpatialJitter.val= nb_dict['spat'][i]
    opt.noise.EnableSpectralJitter.val= nb_dict['spec'][i]   
    opt.background.EnableEmission.val = nb_dict['emm_switch'][i]
    opt.background.EnableZodi.val = nb_dict['zodi_switch'][i]    
    opt.background.EnableDC.val  =  nb_dict['dc_switch'][i]
    opt.background.EnableSource.val  = nb_dict['source_switch'][i]
    opt.diff = nb_dict['diff'][i]      
    opt.noise_tag = nb_dict['noise_tag'][i]
    opt.color = nb_dict['color'][i]
                  
    #==============================================================================
    #  Run with right recipe
    #==============================================================================
    opt.lab = '%s_%s'%(ch, pl)
    opt.no_real = opt.simulation.sim_realisations.val
    opt.diagnostics = opt.simulation.sim_diagnostics.val
    opt.input_params = input_params
    opt.jexosim_path = jexosim_path
    
    jexosim_msg(('Simulation mode %s'%(int(opt.simulation.sim_mode.val))), 1)
          
    if opt.simulation.sim_mode.val == 1:
          recipe  = recipe_1(opt)
    if opt.simulation.sim_mode.val == 2:
       if opt.simulation.sim_output_type.val == 1:   
           recipe  = recipe_2(opt)   
       elif opt.simulation.sim_output_type.val == 2: # fits only
           recipe  = recipe_2a(opt)   
    if opt.simulation.sim_mode.val == 3:
          recipe  = recipe_3(opt)
          
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
    
    run('jexosim_input_params_ex1.txt')      
