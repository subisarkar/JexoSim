"""
JexoSim 
2.0
Exosystem module
v1.0

"""

from jexosim.classes.star    import Star
from jexosim.classes.planet  import Planet
from jexosim.lib.jexosim_lib import jexosim_msg, jexosim_plot
from astropy import units as u
 
class model_star_class():
    def __init__(self, opt):
        self.R = opt.model_exosystem.R_s.val
        self.M = opt.model_exosystem.M_s.val
        self.d = opt.model_exosystem.d.val
        self.T = opt.model_exosystem.T_s.val
        self.Z = opt.model_exosystem.Z.val
        self.logg = opt.model_exosystem.logg.val
        self.name = opt.model_exosystem.star_name.val
        
    def calcLogg(self):
        return self.logg
        
class model_exosystem():
    def __init__(self, opt):
        self.i = opt.model_exosystem.i.val
        self.P = opt.model_exosystem.P.val 
        self.a = opt.model_exosystem.a.val
        self.e = opt.model_exosystem.e.val
        self.R = opt.model_exosystem.R_p.val
        self.M = opt.model_exosystem.M_p.val
        self.T = opt.model_exosystem.T_p.val
        self.name = opt.model_exosystem.planet_name.val
        self.albedo = opt.model_exosystem.albedo.val
        self.star = model_star_class(opt)
        

def run(opt):
  
  opt.cr_path = opt.exosystem_params.planet_spectrum_file().replace('__path__', opt.__path__)
     
  if opt.exosystem_params.planet_use_database.val == 0: # override data base with user defined inputs
          
        opt.model_exosystem.i.val = opt.input_params['user_defined_i']*u.deg
        opt.model_exosystem.P.val = opt.input_params['user_defined_P']*u.day
        opt.model_exosystem.a.val = opt.input_params['user_defined_a']*u.au
        opt.model_exosystem.e.val = opt.input_params['user_defined_e']
        opt.model_exosystem.R_p.val = opt.input_params['user_defined_R_p']*u.Rjup
        opt.model_exosystem.M_p.val = opt.input_params['user_defined_M_p']*u.Mjup
        opt.model_exosystem.T_p.val = opt.input_params['user_defined_T_p']*u.K
        opt.model_exosystem.planet_name.val = opt.input_params['user_defined_planet_name']
        opt.model_exosystem.albedo.val = opt.input_params['user_defined_albedo']
        
        opt.model_exosystem.R_s.val = opt.input_params['user_defined_R_s']*u.Rsun
        opt.model_exosystem.M_s.val = opt.input_params['user_defined_M_s']*u.Msun
        opt.model_exosystem.d.val = opt.input_params['user_defined_d']*u.pc
        opt.model_exosystem.T_s.val = opt.input_params['user_defined_T_s']*u.K
        opt.model_exosystem.Z.val = opt.input_params['user_defined_Z']
        opt.model_exosystem.logg.val = opt.input_params['user_defined_logg']
        opt.model_exosystem.star_name.val = opt.input_params['user_defined_star_name']
           
        opt.model_exosystem.ecliptic_lat.val = opt.input_params['user_defined_ecliptic_lat']*u.deg
        opt.model_exosystem.J_mag.val = opt.input_params['user_defined_J_mag']
        opt.model_exosystem.K_mag.val = opt.input_params['user_defined_K_mag']

        
  jexosim_msg('Exosystem parameters \n---------------------', 1)
  for key in vars(opt.model_exosystem).keys():
        if hasattr(vars(opt.model_exosystem)[key] , 'val'):
            print (key, vars(opt.model_exosystem)[key].val) 
  jexosim_msg('---------------------', 1)
 
  opt.exosystem = model_exosystem(opt) # read in the xml file values
      
  star = Star(opt)
  
  jexosim_msg('exosystem check 1 %s'%(star.sed.sed.max()), opt.diagnostics)  
  
  planet = Planet(opt, opt.exosystem)
  planet.calc_T14((planet.planet.i).to(u.rad),
		 (planet.planet.a).to(u.m), 
		 (planet.planet.P).to(u.s), 
		 (planet.planet.R).to(u.m), 
		 (planet.planet.star.R).to(u.m))             
  
 
  opt.star = star
  opt.planet = planet
  
  return star, planet
