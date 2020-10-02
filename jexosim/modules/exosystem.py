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
 
class fake_star_class():
    def __init__(self, opt):
        self.R = opt.fake_exosystem.R_s.val
        self.M = opt.fake_exosystem.M_s.val
        self.d = opt.fake_exosystem.d.val
        self.T = opt.fake_exosystem.T_s.val
        self.Z = opt.fake_exosystem.Z.val
        self.logg = opt.fake_exosystem.logg.val
        self.name = opt.fake_exosystem.star_name.val
        
    def calcLogg(self):
        return self.logg
        
class fake_exosystem():
    def __init__(self, opt):
        self.i = opt.fake_exosystem.i.val
        self.P = opt.fake_exosystem.P.val 
        self.a = opt.fake_exosystem.a.val
        self.e = opt.fake_exosystem.e.val
        self.R = opt.fake_exosystem.R_p.val
        self.M = opt.fake_exosystem.M_p.val
        self.T = opt.fake_exosystem.T_p.val
        self.name = opt.fake_exosystem.planet_name.val
        self.albedo = opt.fake_exosystem.albedo.val
        self.star = fake_star_class(opt)
        

def run(opt):
  
  opt.cr_path = opt.exosystem_params.planet_spectrum_file().replace('__path__', opt.__path__)
     
  if opt.exosystem_params.planet_use_database.val == 0: # override data base with user defined inputs
          
        opt.fake_exosystem.i.val = opt.input_params['user_defined_i']*u.deg
        opt.fake_exosystem.P.val = opt.input_params['user_defined_P']*u.day
        opt.fake_exosystem.a.val = opt.input_params['user_defined_a']*u.au
        opt.fake_exosystem.e.val = opt.input_params['user_defined_e']
        opt.fake_exosystem.R_p.val = opt.input_params['user_defined_R_p']*u.Rjup
        opt.fake_exosystem.M_p.val = opt.input_params['user_defined_M_p']*u.Mjup
        opt.fake_exosystem.T_p.val = opt.input_params['user_defined_T_p']*u.K
        opt.fake_exosystem.planet_name.val = opt.input_params['user_defined_planet_name']
        opt.fake_exosystem.albedo.val = opt.input_params['user_defined_albedo']
        
        opt.fake_exosystem.R_s.val = opt.input_params['user_defined_R_s']*u.Rsun
        opt.fake_exosystem.M_s.val = opt.input_params['user_defined_M_s']*u.Msun
        opt.fake_exosystem.d.val = opt.input_params['user_defined_d']*u.pc
        opt.fake_exosystem.T_s.val = opt.input_params['user_defined_T_s']*u.K
        opt.fake_exosystem.Z.val = opt.input_params['user_defined_Z']
        opt.fake_exosystem.logg.val = opt.input_params['user_defined_logg']
        opt.fake_exosystem.star_name.val = opt.input_params['user_defined_star_name']
           
        opt.fake_exosystem.ecliptic_lat.val = opt.input_params['user_defined_ecliptic_lat']*u.deg
        
  jexosim_msg('Exosystem parameters \n---------------------', 1)
  for key in vars(opt.fake_exosystem).keys():
        if hasattr(vars(opt.fake_exosystem)[key] , 'val'):
            print (key, vars(opt.fake_exosystem)[key].val) 
  jexosim_msg('---------------------', 1)
 
  opt.exosystem = fake_exosystem(opt) # read in the xml file values
      
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
