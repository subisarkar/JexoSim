"""
JexoSim 
2.0
Exosystem module
v1.0

"""

from jexosim.classes.star    import Star
from jexosim.classes.planet  import Planet
from jexosim.lib.jexosim_lib import jexosim_msg
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
     
  if opt.input_params['planet_use_database'] == 0: # override data base with user defined inputs
          
        opt.fake_exosystem.i.val = opt.input_params['exosystem_i']*u.deg
        opt.fake_exosystem.P.val = opt.input_params['exosystem_P']*u.day
        opt.fake_exosystem.a.val = opt.input_params['exosystem_a']*u.au
        opt.fake_exosystem.e.val = opt.input_params['exosystem_e']
        opt.fake_exosystem.R_p.val = opt.input_params['exosystem_R_p']*u.Rjup
        opt.fake_exosystem.M_p.val = opt.input_params['exosystem_M_p']*u.Mjup
        opt.fake_exosystem.T_p.val = opt.input_params['exosystem_T_p']*u.K
        opt.fake_exosystem.planet_name.val = opt.input_params['exosystem_planet_name']
        opt.fake_exosystem.albedo.val = opt.input_params['exosystem_albedo']
        
        opt.fake_exosystem.R_s.val = opt.input_params['exosystem_R_s']*u.Rsun
        opt.fake_exosystem.M_s.val = opt.input_params['exosystem_M_s']*u.Msun
        opt.fake_exosystem.d.val = opt.input_params['exosystem_d']*u.pc
        opt.fake_exosystem.T_s.val = opt.input_params['exosystem_T_s']*u.K
        opt.fake_exosystem.Z.val = opt.input_params['exosystem_Z']
        opt.fake_exosystem.logg.val = opt.input_params['exosystem_logg']
        opt.fake_exosystem.star_name.val = opt.input_params['exosystem_star_name']
           
        opt.fake_exosystem.ecliptic_lat.val = opt.input_params['exosystem_ecliptic_lat']*u.deg
        

        # print (opt.fake_exosystem.i.val)
        # print (opt.fake_exosystem.P.val)
        # print (opt.fake_exosystem.a.val)
        # print (opt.fake_exosystem.e.val)
        # print (opt.fake_exosystem.R_p.val)
        # print (opt.fake_exosystem.M_p.val)
        # print (opt.fake_exosystem.T_p.val)
        # print (opt.fake_exosystem.planet_name.val)
        # print (opt.fake_exosystem.albedo.val)
        # print (opt.fake_exosystem.R_s.val)
        # print (opt.fake_exosystem.M_s.val)
        # print (opt.fake_exosystem.d.val)
        # print (opt.fake_exosystem.T_s.val)
        # print (opt.fake_exosystem.Z.val)
        # print (opt.fake_exosystem.logg.val)
        # print (opt.fake_exosystem.star_name.val)
        # print (opt.fake_exosystem.use_norm.val)
        # print (opt.fake_exosystem.norm_band.val)               
        # print (opt.fake_exosystem.norm_mag.val)     
        # print (opt.exosystem_params.ecliptic_lat.val) 
      
  
  opt.exosystem = fake_exosystem(opt) # read in the xml file values
      
  star = Star(opt)
  
  jexosim_msg('exosystem check 1 %s'%(star.sed.sed.max()), opt.diagnostics)
  
  star.sed.rebin(opt.common.common_wl)
  jexosim_msg('exosystem check 2 %s'%(star.sed.sed.max()), opt.diagnostics)
  planet = Planet(opt, opt.exosystem)
  planet.calc_T14((planet.planet.i).to(u.rad),
		 (planet.planet.a).to(u.m), 
		 (planet.planet.P).to(u.s), 
		 (planet.planet.R).to(u.m), 
		 (planet.planet.star.R).to(u.m))             
                   
  if opt.timeline.apply_lc.val ==1:  
      planet.sed.rebin(opt.common.common_wl) 


  opt.star = star
  opt.planet = planet
  
  return star, planet
