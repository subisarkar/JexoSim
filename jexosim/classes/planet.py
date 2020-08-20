"""
JexoSim 
2.0
Planet class
v1.0

"""


import numpy as np
from jexosim.classes import sed
from astropy import units as u
from jexosim.lib.jexosim_lib import jexosim_msg, jexosim_plot
from pytransit import QuadraticModel 

class Planet(object):
  """
    Instantiate a Planet class 

  """
  def __init__(self, opt, exosystem):
      
    self.planet = exosystem
    self.opt =opt
    
    if opt.exosystem_params.use_planet_spectrum_file.val ==1:
        _d = np.loadtxt(opt.cr_path)
        pl_wl, pl_sed=  _d[:,0]*u.um, _d[:,1]*u.dimensionless_unscaled
        idx = np.argsort(pl_wl)
        self.sed = sed.Sed(pl_wl[idx], pl_sed[idx])           
  
    else:
        self.simple_model() 
        jexosim_msg ("simple planet spectrum chosen with fixed CR of", self.sed.sed[0])
          
    
  def simple_model(self):    #need to complete    
    if self.opt.exosystem_params.primary_transit.val == 1:
        wl = np.arange(0.3,30,0.1)*u.um
        cr = np.array([( (self.planet.R).to(u.m)/(self.planet.star.R).to(u.m))**2]*len(wl))*u.dimensionless_unscaled
        self.sed = sed.Sed(wl,cr)      
    else:
        self.sed = flux_ratio #tbd
                
  def calc_T14(self, inc, a, per, Rp, Rs):
    b = np.cos(inc)*a/Rs
    rat = Rp/Rs
    self.t14 = per/np.pi* Rs/a * np.sqrt((1+rat)**2 - b**2)
    return self.t14
         
  def get_light_curve(self, opt, planet_sed, wavelength, transit_type):
      
    wavelength = wavelength.value
    
    timegrid = opt.z_params[0]
    t0 = opt.z_params[1]
    per = opt.z_params[2]
    ars = opt.z_params[3]
    inc = opt.z_params[4]
    ecc = opt.z_params[5]
    omega = opt.z_params[6]
       
    if opt.timeline.useLDC.val == 0:   # linear ldc
        u0 = np.zeros(len(wavelength)); u1 = np.zeros(len(wavelength))
        ldc = np.vstack((wavelength,u0,u1))

    elif opt.timeline.useLDC.val == 1:   # use to get ldc from filed values
        u0,u1 = self.getLDC_interp(self.planet, wavelength)
        ldc = np.vstack((wavelength,u0,u1))          

    elif opt.timeline.useLDC.val == 3:  # use to calcaulte ldc in real time with exotethys
        self.ldCoeffs = self.getLDC()        
        u0, u1 =  self.ldCoeffs_new.getCoeffs(self.planet, opt.x_wav_edges) 
        wavelength = opt.x_wav
        if wavelength.max() > 9.9: # fix since no LDCs calced above 9.9 microns -assume no changes above this
          idx = np.argwhere(wavelength>9.9)
          u0[idx] = u0[idx[-1].item()+1]
          u1[idx] = u1[idx[-1].item()+1]
        u = np.vstack((u0,u1))
        u = u.T   
        ldc = np.vstack((wavelength,u[:,0],u[:,1])) 
        
    elif opt.timeline.useLDC.val == 2:  #preloaded LDCS for tests - ignore
          if opt.channel.name == 'MIRI_LRS':
              aa = np.load('/Users/user1/Desktop/aa_MIRI.npy')
              
          elif opt.channel.name ==  'NIRSpec_G395M':
              aa = np.load('/Users/user1/Desktop/aa_NS.npy')
          u = np.zeros((aa.shape[1], 2))
          u[:,0] = aa[1]
          u[:,1] = aa[2]
          ldc = aa   
 
             
    u = np.zeros((ldc.shape[1],2))
    u[:,0] = ldc[1]
    u[:,1] = ldc[2]
    
    jexosim_plot('ldc', opt.diagnostics, ydata =ldc[1])
    jexosim_plot('ldc', opt.diagnostics, ydata =ldc[2])
 
    
      
    tm = QuadraticModel(interpolate=False)
    tm.set_data(timegrid)
 
                                           
    lc = np.zeros((len(planet_sed), len(timegrid) ))
    
    if transit_type == 1: #primary transit
        for i in range(len(planet_sed)):
   
              k = np.sqrt(planet_sed[i]).value      
              lc[i, ...]= tm.evaluate(k=k, ldc=u[i, ...], t0=t0, p=per, a=ars, i=inc, e=ecc, w=omega)   
              
              jexosim_plot('light curve check', opt.diagnostics, ydata = lc[i, ...] )
          
 
    elif transit_type == 2: # secondary eclipse
        for i in range(len(planet_sed)):
              k = np.sqrt(planet_sed[i])  # planet star radius ratio 
              lc_base = tm.evaluate(k=k, ldc=[0,0], t0=t0, p=per, a=ars, i=inc, e=ecc, w=omega)                      
              f_e = (planet_sed[i] + (   lc_base  -1.0))/planet_sed[i]
              lc[i, ...]  = 1.0 + f_e*(planet_sed[i])  

    
    return lc, ldc
          
          
  def getLDC_direct(self, planet, wl):
        
          jexosim_msg("Calculating LDC directly from exothethys", 1)
   
          import pickle
          from exotethys import sail
          import matplotlib.pyplot as plt
        
          T_s = planet.star.T.value          
          log_g = planet.star.logg
          jexosim_msg ("T_s, logg>>>>  %s %s"%(T_s, log_g  ), self.opt.diagnostics)

          wl0= wl*1        
          wl = (wl[::-1].value)*1e4  
          for i in range (len(wl)):
              if wl[i] >0:
                  idx1 = i
                  break
          for i in range (1,len(wl)):
              if wl[-i] >0:
                  idx2 = len(wl0) - i
                  break

          idx1 =idx1+1 # fix to deal with issue in MIRI - double wavelength appears at start of list messing interp
          wl = wl[idx1:idx2+1]
          for i in range (1,len(wl)):
              if wl[-i] < 99000:
                  idx3 = len(wl) - i
                  break                      
          wl = wl[:idx3+1]
    
          b = '%s/Wavelength_bins_files/Wavelength_bins.txt'%(self.opt.exotethys_path)  
          open(b, 'w').close()      
          file = open(b,'w') 
          for q in range (len(wl)-1):
                file.write(str(wl[q])+'     '+str(wl[q+1])+'\n')
          file.close()          
                   
          b = '%s/examples/input_ldc.txt'%(self.opt.exotethys_path) 
     
          open(b, 'w').close() 
          file = open(b,'w')         
          file.write("##GENERIC \n")
          file.write("calculation_type			!grid		individual \n")
          file.write("stellar_models_grid			!Phoenix_2018	Phoenix_2012_13		!Atlas \n")
          file.write("limb_darkening_laws			claret4		!power2		!square_root	    quadratic	!linear		!gen_claret	!gen_poly \n")
          file.write("passbands				uniform_phoenix_2012_13.pass \n")
          file.write("wavelength_bins_files			Wavelength_bins.txt \n")
          file.write("user_output				basic		!complete \n")
          file.write("output_path                  . \n")
          file.write("target_names				Star \n")	 	
          file.write("star_effective_temperature		%s \n"%(T_s))	 
          file.write("star_log_gravity			%s \n"%(log_g))	 
          file.write("star_metallicity			0.0 \n")
          file.close()
                
          sail.ldc_calculate(b)              
          a = '%s/Star_ldc.pickle'%(self.opt.exotethys_path)                              
          a = pickle.load(open(a, 'rb'))
                 #          print a.keys()
                    #          print a['star_params']
                    #          print a['passbands'].keys()
                    #            
                    #          for i in a['passbands'].keys():
                    #                print i, a['passbands'][i]['quadratic']['coefficients']
                    
                    #xxxx  
                             
          wl_list1 = []
          wl_list2 = []
          u1_list = []
          u2_list = []
          for i in a['passbands'].keys():
                
            #    jexosim_msg i, a['passbands'][i]['quadratic']['coefficients']
                u1 =  a['passbands'][i]['quadratic']['coefficients'][0]
                u2 =  a['passbands'][i]['quadratic']['coefficients'][1]
                r = i[24:]
                
                idx = r.find('_')
                r1 = np.float(r[:idx])
                r2 = np.float(r[idx+1:])
                
                wl_list1.append(r1)
                wl_list2.append(r2) 
              
                u1_list.append(u1)
                u2_list.append(u2)
            #                print r1, r2, u1
          ll = zip(wl_list1,wl_list2, u1_list,u2_list)
          ll.sort() 
          wl_list1 = [x[0] for x in ll]
          wl_list2 = [x[1] for x in ll]   
          u1_list = [x[2] for x in ll]   
          u2_list = [x[3] for x in ll]
          
          jexosim_plot('ldcs', self.opt.diagnostics, xdata= wl_list1, ydata=u1_list)
          jexosim_plot('ldcs', self.opt.diagnostics, xdata= wl_list1, ydata=u2_list)
      
            
          u1 = np.zeros(len(wl0))
          u2 = np.zeros(len(wl0))
          u1[idx1:idx1+ len(u1_list)] = u1_list
          u2[idx1:idx1+ len(u1_list)] = u2_list
          u1 = u1[::-1][1:]
          u2 = u2[::-1][1:]
       
          return u1, u2
 

  def getLDC_interp(self, planet, wl):
          
        jexosim_msg ('Calculating LDC from pre-installed files', 1)

        T = planet.star.T.value
        if T < 3000.0:  # temp fix as LDCs not available for T< 3000 K
            T=3000.0
        logg = planet.star.logg
        jexosim_msg ("T_s, logg>>>> %s %s"%(T, logg ), self.opt.diagnostics)
      
        folder = '%s/data/LDC'%(self.opt.__path__)
        
        t_base = np.int(T/100.)*100
        if T>=t_base+100:
            t1 = t_base+100
            t2 = t_base+200
        else:
            t1 = t_base 
            t2 = t_base+100 
            
        logg_base = np.int(logg)   
        if logg>= logg_base+0.5:
            logg1 = logg_base+0.5
            logg2 = logg_base+1.0
        else:
            logg1 = logg_base 
            logg2 = logg_base+0.5
        logg1 = np.float(logg1)
        logg2 = np.float(logg2)
 
        
        #==============================================================================
        # interpolate between temperatures assuming logg1
        #==============================================================================
        
        u0_a = np.loadtxt('%s/T=%s_logg=%s_z=0.txt'%(folder,t1, logg1))[:,1]
        u0_b = np.loadtxt('%s/T=%s_logg=%s_z=0.txt'%(folder,t2, logg1))[:,1]
        w2 = 1.0-abs(t2-T)/100.
        w1 = 1.0-abs(t1-T)/100.
        u0_1 = w1*u0_a + w2*u0_b
        
        u1_a = np.loadtxt('%s/T=%s_logg=%s_z=0.txt'%(folder,t1, logg1))[:,2]
        u1_b = np.loadtxt('%s/T=%s_logg=%s_z=0.txt'%(folder,t2, logg1))[:,2]
        w2 = 1.0-abs(t2-T)/100.
        w1 = 1.0-abs(t1-T)/100.
        u1_1 = w1*u1_a + w2*u1_b
        
        jexosim_plot('ldc interp logg1', self.opt.diagnostics, ydata=u0_a, marker='b-')
        jexosim_plot('ldc interp logg1', self.opt.diagnostics, ydata=u0_b, marker='r-')   
        jexosim_plot('ldc interp logg1', self.opt.diagnostics, ydata=u0_1, marker='g-')
        jexosim_plot('ldc interp logg1', self.opt.diagnostics, ydata=u1_a, marker='b-')
        jexosim_plot('ldc interp logg1', self.opt.diagnostics, ydata=u1_b, marker='r-')
        jexosim_plot('ldc interp logg1', self.opt.diagnostics, ydata=u1_1, marker='g-')
    
        
        ldc_wl = np.loadtxt('%s/T=%s_logg=%s_z=0.txt'%(folder,t1, logg1))[:,0]
        
        #==============================================================================
        # interpolate between temperatures assuming logg2
        #==============================================================================
        
        u0_a = np.loadtxt('%s/T=%s_logg=%s_z=0.txt'%(folder,t1, logg2))[:,1]
        u0_b = np.loadtxt('%s/T=%s_logg=%s_z=0.txt'%(folder,t2, logg2))[:,1]
        w2 = 1.0-abs(t2-T)/100.
        w1 = 1.0-abs(t1-T)/100.
        u0_2 = w1*u0_a + w2*u0_b
        
        u1_a = np.loadtxt('%s/T=%s_logg=%s_z=0.txt'%(folder,t1, logg2))[:,2]
        u1_b = np.loadtxt('%s/T=%s_logg=%s_z=0.txt'%(folder,t2, logg2))[:,2]
        w2 = 1.0-abs(t2-T)/100.
        w1 = 1.0-abs(t1-T)/100.
        u1_2 = w1*u1_a + w2*u1_b
        
        
        jexosim_plot('ldc interp logg2', self.opt.diagnostics, ydata=u0_a, marker='b-')
        jexosim_plot('ldc interp logg2', self.opt.diagnostics, ydata=u0_b, marker='r-')   
        jexosim_plot('ldc interp logg2', self.opt.diagnostics, ydata=u0_2, marker='g-')
        jexosim_plot('ldc interp logg2', self.opt.diagnostics, ydata=u1_a, marker='b-')
        jexosim_plot('ldc interp logg2', self.opt.diagnostics, ydata=u1_b, marker='r-')
        jexosim_plot('ldc interp logg2', self.opt.diagnostics, ydata=u1_2, marker='g-')

        
        #==============================================================================
        # interpolate between logg
        #==============================================================================
        w2 = 1.0-abs(logg2-logg)/.5
        w1 = 1.0-abs(logg1-logg)/.5
        
        u0 = w1*u0_1 + w2*u0_2
        u1 = w1*u1_1 + w2*u1_2

        jexosim_plot('ldc interp', self.opt.diagnostics, ydata=u0_1, marker='b-')
        jexosim_plot('ldc interp', self.opt.diagnostics, ydata=u0_2, marker='r-')   
        jexosim_plot('ldc interp', self.opt.diagnostics, ydata=u0, marker='g-')
        jexosim_plot('ldc interp', self.opt.diagnostics, ydata=u1_1, marker='b-')
        jexosim_plot('ldc interp', self.opt.diagnostics, ydata=u1_2, marker='r-')
        jexosim_plot('ldc interp', self.opt.diagnostics, ydata=u1, marker='g-')
            

        #==============================================================================
        # interpolate to new wl grid
        #==============================================================================
    
        u0_final = np.interp(wl,ldc_wl,u0)
        u1_final = np.interp(wl,ldc_wl,u1)
  
        jexosim_plot('ldc final', self.opt.diagnostics, xdata = ldc_wl,  ydata=u0, marker='ro', alpha=0.1)
        jexosim_plot('ldc final', self.opt.diagnostics, xdata = ldc_wl,  ydata=u1, marker='bo', alpha=0.1)
        jexosim_plot('ldc final', self.opt.diagnostics, xdata = wl,  ydata=u0_final, marker='rx')
        jexosim_plot('ldc final', self.opt.diagnostics, xdata = wl,  ydata=u1_final, marker='bx')
 

        
        return u0_final, u1_final