"""
JexoSim 
2.0
Exosystem module
v1.0

"""

from jexosim.classes.star    import Star
from jexosim.classes.planet  import Planet
from jexosim.lib.jexosim_lib import jexosim_msg, jexosim_plot
from jexosim.lib import jexosim_lib, exosystem_lib
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
        self.T14 = opt.model_exosystem.T14.val
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
        if 'user_defined_T14' in opt.input_params.keys():
            opt.model_exosystem.T14.val = opt.input_params['user_defined_T14']*u.hr
        else:
            pl_inc = (opt.input_params['user_defined_i']*u.deg).to(u.rad)
            pl_a  =(opt.input_params['user_defined_a']*u.au).to(u.m)
            pl_P  =(opt.input_params['user_defined_P']*u.day).to(u.hr)
            pl_Rp = (opt.input_params['user_defined_R_p']*u.Rjup).to(u.m)
            pl_Rs = (opt.input_params['user_defined_R_s']*u.Rsun).to(u.m)         
            T14 = exosystem_lib.calc_T14(pl_inc, pl_a, pl_P, pl_Rp, pl_Rs)
            T14 = T14
            jexosim_msg(f'No user defined T14: calculating T14 estimate...{T14}', opt.diagnostics)
        opt.model_exosystem.logg.val = opt.input_params['user_defined_logg']
        opt.model_exosystem.star_name.val = opt.input_params['user_defined_star_name']
           
        opt.model_exosystem.ecliptic_lat.val = opt.input_params['user_defined_ecliptic_lat']*u.deg
        opt.model_exosystem.J_mag.val = opt.input_params['user_defined_J_mag']
        opt.model_exosystem.K_mag.val = opt.input_params['user_defined_K_mag']
        
        if 'user_defined_ra' in opt.input_params.keys():
            opt.model_exosystem.ra.val = opt.input_params['ra']
        else:
            opt.model_exosystem.ra.val = 0
         
        if 'user_defined_dec' in opt.input_params.keys():
            opt.model_exosystem.dec.val = opt.input_params['ra']
        else:
            opt.model_exosystem.dec.val = 0          
         
        
        if  opt.input_params['user_defined_logg'] == '':
            opt.model_exosystem.logg.val = jexosim_lib.calc_logg(opt.model_exosystem.M_s.val,opt.model_exosystem.R_s.val)[1]

        
  jexosim_msg('Exosystem parameters \n---------------------', 1)
  for key in vars(opt.model_exosystem).keys():
        if hasattr(vars(opt.model_exosystem)[key] , 'val'):
            print (key, vars(opt.model_exosystem)[key].val) 
  jexosim_msg('---------------------', 1)
 
  opt.exosystem = model_exosystem(opt) # read in the xml file values
      
  opt.star = Star(opt)
  
  jexosim_msg('exosystem check 1 %s'%(opt.star.sed.sed.max()), opt.diagnostics)  
  
  opt.planet = Planet(opt, opt.exosystem)
  opt.planet.T14 = opt.model_exosystem.T14.val
  
  # obtain the in-transit stellar spectrum at high resolution
  # rebin star or planet sed to that of the lower resolution spec

             
  jexosim_msg('T14 %s'%(opt.planet.T14), 1)
  # xxxxx
   
  opt  = exosystem_lib.get_it_spectrum(opt) 
  
  return opt
  # return star, planet
