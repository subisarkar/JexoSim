"""
JexoSim 
2.0
Planet class
v1.1

"""


import numpy as np
from jexosim.classes import sed
from astropy import units as u
from astropy import constants as const
from jexosim.lib.jexosim_lib import jexosim_msg, jexosim_plot, planck
import os, sys

class Planet(object):
  """
    Instantiate a Planet class 

  """
  def __init__(self, opt, exosystem):
      
    self.planet = exosystem
    self.opt =opt
    
    if opt.exosystem_params.planet_spectrum_model.val =='complex':
        self.complex_model()
        jexosim_msg ("complex planet spectrum chosen from database", 1)           
    elif opt.exosystem_params.planet_spectrum_model.val == 'simple':
        self.simple_model() 
        jexosim_msg ("simple planet spectrum chosen with fixed CR of %s"%(self.sed.sed[0]), 1)
    elif opt.exosystem_params.planet_spectrum_model.val =='file':
        self.file_model()  
        jexosim_msg ("filed planet spectrum chosen", 1)
    else:
        jexosim_msg('Error1 : no compatible entry for planet_spectrum_model', 1)
        sys.exit()
          
    
  def simple_model(self):    #need to complete    
    if self.opt.observation.obs_type.val == 1:
        wl = np.arange(0.3,30,0.1)*u.um
        cr = np.array([( (self.planet.R).to(u.m)/(self.planet.star.R).to(u.m))**2]*len(wl))*u.dimensionless_unscaled
        self.sed = sed.Sed(wl,cr)      
    elif self.opt.observation.obs_type.val == 2:
        wl = np.arange(0.3,30,0.1)*u.um
        star_flux =  np.pi*planck(wl, self.planet.star.T)*(  (self.planet.star.R).to(u.m) / (self.planet.star.d).to(u.m))**2
        planet_flux_em =  np.pi*planck(wl, self.planet.T)*(  (self.planet.R).to(u.m) / (self.planet.star.d).to(u.m))**2
        planet_flux_ref =  star_flux * (self.planet.albedo / 4.) *(  (self.planet.R).to(u.m) / (self.planet.a).to(u.m))**2
        planet_flux = planet_flux_em + planet_flux_ref 
        cr = planet_flux / star_flux
        self.sed = sed.Sed(wl,cr)   
        # import matplotlib.pyplot as plt
        # plt.plot(wl, planet_flux)
        # plt.plot(wl, star_flux)
        # xxxx
    else:
        jexosim_msg("Error2 : no compatible entry for obs_type", 1)
        sys.exit()


  def complex_model(self):    #need to complete    
      if self.opt.observation.obs_type.val == 1:
          Tp =  self.planet.T
          Rs =  self.planet.star.R # in Rsol units
          Rp =  self.planet.R # in Rjup units
          Mp =  self.planet.M

          # Tp = 1500*u.K
          # Rs=1*u.Rsun
          # Rp=1*u.Rjup
          # Mp =1*u.Mjup
        
          Gp =  self.calc_gravity(Mp,Rp)
          
          met = self.opt.exosystem_params.planet_spectrum_params_met.val
          co = self.opt.exosystem_params.planet_spectrum_params_co.val
          haze = self.opt.exosystem_params.planet_spectrum_params_haze.val
          cloud = self.opt.exosystem_params.planet_spectrum_params_cloud.val
          cond = self.opt.exosystem_params.planet_spectrum_params_cond.val
          mmw = self.opt.exosystem_params.planet_spectrum_params_mmw.val
          
          if cond == 'local':
              target_folder = 'local'
          elif cond == 'rainout':
              target_folder = 'rainout'
          else:
              jexosim_msg("Error: no compatible entry for planet_spectrum_params_cond", 1)
              sys.exit()
    
          databases_dir = '%s/archive'%(self.opt.jexosim_path)
          cond=0
          for root, dirs, files in os.walk(databases_dir):
              for dirc in dirs:
                if target_folder in dirc:
                    dirc_name = dirc
                    cond=1
                    break
          if cond==0:
              print ('Error: planet spectrum database not found')    
          folder = '%s/%s'%(databases_dir, dirc_name)
          
          t = str(int(np.round(Tp.value/100.)*100))
          l = len(t)
          while l < 4:    
              t = '0'+t
              l = len(t)
          t0 = t    
 
          met_list = [2.3, 2.0, 1.7, 1, 0, -1]
          met_str_list = ['+2.3', '+2.0', '+1.7', '+1.0', '+0.0','-1.0']
          g_list = [50, 20, 10, 5]
          g_str_list = ['50', '20', '10', '05']
          co_list = [1.0, 0.7, 0.56, 0.35]
          co_str_list = ['1.00', '0.70', '0.56', '0.35']
          haze_list = [1100.0, 150.0, 10.0, 1.0]
          haze_str_list = ['1100', '0150', '0010', '0001']
          cloud_list = [1.0, 0.2, 0.06, 0.0]
          cloud_str_list = ['1.00', '0.20', '0.06', '0.00']
          mmw_list = [5.5, 4.0, 3.2, 2.5, 2.3, 2.3] 

          def input_val(val, list0, str_list):
             diff =[]
             for i in range(len(list0)):
                diff.append(np.abs(val-list0[i]))
             diff = np.array(diff)
             idx = np.argmin(diff)
             return str_list[idx]

          met0= (input_val(met, met_list, met_str_list))
          g0 = (input_val(Gp.value, g_list, g_str_list))
          co0 = (input_val(co, co_list, co_str_list))
          haze0= (input_val(haze, haze_list, haze_str_list))
          cloud0=  (input_val(cloud, cloud_list, cloud_str_list))

          filename_root = 'trans-iso-generic_%s_%s_%s_%s_%s_%s_model'%(t0,g0,met0,co0,haze0,cloud0)         
          # filename_root = 'trans-iso-generic_1500_10_+0.0_1.00_0001_0.00_model'
          print (filename_root)
          
          idx = np.argwhere(np.array(met_str_list) == met0)[0].item()
          mmw = mmw_list[idx]

          wl, cr = self.spectrum_rescale(folder, filename_root, Rs, Rp, Gp, Tp, mmw)
          wl = wl*u.um
          cr = cr*u.dimensionless_unscaled
          self.sed = sed.Sed(wl,cr)  
          # import matplotlib.pyplot as plt
          # plt.figure(4)
          # plt.plot(wl, cr)
          # xxx
       
      elif self.opt.observation.obs_type.val == 2: # currently no grid for this
          wl = np.arange(0.3,30,0.1)*u.um
          star_flux =  np.pi*planck(wl, self.planet.star.T)*(  (self.planet.star.R).to(u.m) / (self.planet.star.d).to(u.m))**2
          planet_flux_em =  np.pi*planck(wl, self.planet.T)*(  (self.planet.R).to(u.m) / (self.planet.star.d).to(u.m))**2
          planet_flux_ref =  star_flux * (self.planet.albedo / 4.) *(  (self.planet.R).to(u.m) / (self.planet.a).to(u.m))**2
          planet_flux = planet_flux_em + planet_flux_ref 
          cr = planet_flux / star_flux
          self.sed = sed.Sed(wl,cr)    
      else:
          jexosim_msg("Error3: no compatible entry for obs_type", 1)       
  
   
  def file_model(self):
      filename = self.opt.exosystem_params.planet_spectrum_file.val       
      try:
          aa = np.loadtxt(filename)
      except IOError:
          jexosim_msg("Error: No spectrum file found",  1) 
      wl=  aa[:,0]*u.um, 
      cr = aa[:,1]
      self.sed = sed.Sed(wl,cr)   

             
  def calc_T14(self, inc, a, per, Rp, Rs):
    b = np.cos(inc)*a/Rs
    rat = Rp/Rs
    self.t14 = per/np.pi* Rs/a * np.sqrt((1+rat)**2 - b**2)
    return self.t14
         
  def calc_gravity(self, Mp, Rp):
    Mp = 1*u.M_earth
    Rp = 1*u.R_earth
    
    Mp = Mp.to(u.kg)
    Rp = Rp.to(u.m)
    G = const.G
    g  = G*Mp/(Rp**2)
    return g 
 
  def spectrum_rescale(self, folder, filename_root, Rs, Rp, Gp, Tp, mmw):
    """Code adapted from J. Goyal https://drive.google.com/drive/folders/14TX4WlcoayUMHeX-LV9hz-CHkkr0q9F7 
	(Goyal J. et al. (2019) MNRAS 482, 4503–4513) """ 
    
    if os.path.exists('%s/%s.txt'%(folder, filename_root)):
        filename = '%s.txt'%(filename_root)
    elif os.path.exists('%s/%s.txt.gz'%(folder, filename_root)):
        filename = '%s.txt.gz'%(filename_root)
    else:
        jexosim_msg('Error: no planet spectrum file found', 1)
        sys.exit()      
    jexosim_msg('Planets spectrum file selected: %s'%(filename), 1)   
      
    kb = const.k_B.value*u.m**2*u.kg/u.s**2/u.K
    kb = kb.to(u.cm**2*u.g/u.s**2/u.K).value        
    mu = mmw *const.m_p.to(u.g).value
    tau = 0.56  
    rstar = Rs.to(u.cm).value 
    rpl = Rp.to(u.cm).value 
    rp2 = rpl
    rsun = (1*u.Rsun).to(u.cm).value 
    
    model_wav, model_rp = np.loadtxt('%s/%s'%(folder,filename), unpack=True) 
    t1 = float(filename.split('_')[1]) 
    g1 = 100*float(filename.split('_')[2]) 
    h1 = (kb * t1) / (mu * g1) 

    # rp1 = np.sqrt(model_rp) * rsun  
    # z1 = rp1 - (np.sqrt(model_rp[2000])*rsun)  
    
    r1 = np.sqrt(model_rp) * rsun
    rp1 = (1*u.Rjup).to(u.cm).value #cm
    z1 = r1 - rp1 #cm
    
    epsig1 = tau * np.sqrt((kb * t1 * mu * g1) / (2. * np.pi * rp1)) * np.exp(z1 / h1)
    h2 = (kb * Tp.value) / (mu * Gp.to(u.cm/u.s**2).value)

    z2 = h2 * np.log(epsig1 / tau * np.sqrt((2. * np.pi * rp2)/(kb * Tp.value * mu * Gp.to(u.cm/u.s**2).value)))
    r2 = z2 + rp2 
    srt = np.argsort(model_wav) 
    wav = model_wav[srt] 
    r2 = r2[srt] 
    depth = (r2 / rstar)**2 
    
    return wav, depth
