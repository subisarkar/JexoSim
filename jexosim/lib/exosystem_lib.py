"""
JexoSim 2.0

Exosystem and related library

"""

import numpy as np
from jexosim.lib import jexosim_lib
from jexosim.lib.jexosim_lib import jexosim_msg, jexosim_plot
from jexosim.classes.sed import Sed
from pytransit import QuadraticModel 
import copy

import matplotlib.pyplot as plt

def get_light_curve(opt, planet_sed, wavelength, obs_type):
      
    wavelength = wavelength.value    
    timegrid = opt.z_params[0]
    t0 = opt.z_params[1]
    per = opt.z_params[2]
    ars = opt.z_params[3]
    inc = opt.z_params[4]
    ecc = opt.z_params[5]
    omega = opt.z_params[6]
    
    if opt.observation.obs_type.val == 2:
        opt.timeline.useLDC.val = 0
        
    if opt.timeline.useLDC.val == 0:   # linear ldc
        jexosim_msg('LDCs set to zero', 1)
        u0 = np.zeros(len(wavelength)); u1 = np.zeros(len(wavelength))
        ldc = np.vstack((wavelength,u0,u1))
    elif opt.timeline.useLDC.val == 1:   # use to get ldc from filed values
        u0,u1 = getLDC_interp(opt, opt.planet.planet, wavelength)
        ldc = np.vstack((wavelength,u0,u1))

    gamma = np.zeros((ldc.shape[1],2))
    gamma[:,0] = ldc[1]
    gamma[:,1] = ldc[2]    
    jexosim_plot('ldc', opt.diagnostics, ydata =ldc[1])
    jexosim_plot('ldc', opt.diagnostics, ydata =ldc[2])      
    tm = QuadraticModel(interpolate=False)
    tm.set_data(timegrid)
                                            
    lc = np.zeros((len(planet_sed), len(timegrid) ))
    
    if obs_type == 1: #primary transit
        for i in range(len(planet_sed)):   
              k = np.sqrt(planet_sed[i]).value      
              lc[i, ...]= tm.evaluate(k=k, ldc=gamma[i, ...], t0=t0, p=per, a=ars, i=inc, e=ecc, w=omega)                 
              jexosim_plot('light curve check', opt.diagnostics, ydata = lc[i, ...] )    
    elif obs_type == 2: # secondary eclipse
        for i in range(len(planet_sed)):
              k = np.sqrt(planet_sed[i])  # planet star radius ratio 
              lc_base = tm.evaluate(k=k, ldc=[0,0], t0=t0, p=per, a=ars, i=inc, e=ecc, w=omega)
              jexosim_plot('light curve check', opt.diagnostics, ydata = lc_base )                      
              # f_e = (planet_sed[i] + (   lc_base  -1.0))/planet_sed[i]
              # lc[i, ...]  = 1.0 + f_e*(planet_sed[i])  
   
              lc[i,...] =  lc_base + planet_sed[i]
              jexosim_plot('light curve check', opt.diagnostics, ydata = lc[i, ...] )
 
    
    return lc, ldc
 
     
def getLDC_interp(opt, planet, wl):
          
        jexosim_msg ('Calculating LDC from pre-installed files', 1)
        T = planet.star.T.value
        if T < 3000.0:  # temp fix as LDCs not available for T< 3000 K
            T=3000.0
        logg = planet.star.logg
        jexosim_msg ("T_s, logg>>>> %s %s"%(T, logg ), opt.diagnostics)
        folder = '%s/archive/LDC'%(opt.jexosim_path)        
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
        
        jexosim_plot('ldc interp logg1', opt.diagnostics, ydata=u0_a, marker='b-')
        jexosim_plot('ldc interp logg1', opt.diagnostics, ydata=u0_b, marker='r-')   
        jexosim_plot('ldc interp logg1', opt.diagnostics, ydata=u0_1, marker='g-')
        jexosim_plot('ldc interp logg1', opt.diagnostics, ydata=u1_a, marker='b-')
        jexosim_plot('ldc interp logg1', opt.diagnostics, ydata=u1_b, marker='r-')
        jexosim_plot('ldc interp logg1', opt.diagnostics, ydata=u1_1, marker='g-')
            
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
                
        jexosim_plot('ldc interp logg2', opt.diagnostics, ydata=u0_a, marker='b-')
        jexosim_plot('ldc interp logg2', opt.diagnostics, ydata=u0_b, marker='r-')   
        jexosim_plot('ldc interp logg2', opt.diagnostics, ydata=u0_2, marker='g-')
        jexosim_plot('ldc interp logg2', opt.diagnostics, ydata=u1_a, marker='b-')
        jexosim_plot('ldc interp logg2', opt.diagnostics, ydata=u1_b, marker='r-')
        jexosim_plot('ldc interp logg2', opt.diagnostics, ydata=u1_2, marker='g-')        
        #==============================================================================
        # interpolate between logg
        #==============================================================================
        w2 = 1.0-abs(logg2-logg)/.5
        w1 = 1.0-abs(logg1-logg)/.5
    
        u0 = w1*u0_1 + w2*u0_2
        u1 = w1*u1_1 + w2*u1_2

        jexosim_plot('ldc interp', opt.diagnostics, ydata=u0_1, marker='b-')
        jexosim_plot('ldc interp', opt.diagnostics, ydata=u0_2, marker='r-')   
        jexosim_plot('ldc interp', opt.diagnostics, ydata=u0, marker='g-')
        jexosim_plot('ldc interp', opt.diagnostics, ydata=u1_1, marker='b-')
        jexosim_plot('ldc interp', opt.diagnostics, ydata=u1_2, marker='r-')
        jexosim_plot('ldc interp', opt.diagnostics, ydata=u1, marker='g-')        
        #==============================================================================
        # interpolate to new wl grid
        #==============================================================================    
        u0_final = np.interp(wl,ldc_wl,u0)
        u1_final = np.interp(wl,ldc_wl,u1)
  
        jexosim_plot('ldc final', opt.diagnostics, xdata = ldc_wl,  ydata=u0, marker='ro', alpha=0.1)
        jexosim_plot('ldc final', opt.diagnostics, xdata = ldc_wl,  ydata=u1, marker='bo', alpha=0.1)
        jexosim_plot('ldc final', opt.diagnostics, xdata = wl,  ydata=u0_final, marker='rx')
        jexosim_plot('ldc final', opt.diagnostics, xdata = wl,  ydata=u1_final, marker='bx')
       
        return u0_final, u1_final


def calc_T14(inc, a, per, Rp, Rs):
    b = np.cos(inc)*a/Rs
    rat = Rp/Rs
    t14 = per/np.pi* Rs/a * np.sqrt((1+rat)**2 - b**2)
    return t14


def get_it_spectrum(opt):
    
    wl_lo = opt.channel.pipeline_params.wavrange_lo.val
    wl_hi = opt.channel.pipeline_params.wavrange_hi.val
    wl = opt.star.sed.wl
    planet_wl = opt.planet.sed.wl

    R = wl/np.gradient((wl))
    R_planet = planet_wl/np.gradient((planet_wl))
    # select onyl wavelengths in range for channel
    idx = np.argwhere((wl.value>= wl_lo) & (wl.value<= wl_hi)).T[0]
    idx2 = np.argwhere((planet_wl.value>= wl_lo) & (planet_wl.value<= wl_hi)).T[0]
    # rebin to lower resolution spectrum
    if  R_planet[idx2].min() < R[idx].min():
        opt.star.sed.rebin(planet_wl)
    else:
        opt.planet.sed.rebin(wl)
    # now make in-transit stellar spectrum
    star_sed_it = opt.star.sed.sed*(1-opt.planet.sed.sed)
    opt.star.sed_it = Sed(opt.star.sed.wl, star_sed_it)
 
    return opt

