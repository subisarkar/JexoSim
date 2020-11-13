"""
JexoSim
2.0
Systematics module
v1.01
    
    """
import numpy as np
from jexosim.lib import jexosim_lib
from jexosim.lib.jexosim_lib import jexosim_msg
import copy
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d



def add_systematic_grid(opt):
    
    wl = opt.x_wav_osr[1::3]
    time = opt.ndr_end_time
    
    syst_grid_path = opt.simulation.sim_systematic_model_path.val
    syst_grid0 = np.loadtxt(syst_grid_path)
    
    idx = syst_grid_path.find('systematic_models')+18
    syst_wl =  np.loadtxt('%s%s'%(syst_grid_path[:idx], 'wl.txt'))
    syst_time =  np.loadtxt('%s%s'%(syst_grid_path[:idx], 'time.txt'))
    
    
    f = interp2d(syst_time, syst_wl, syst_grid0, kind='linear')
    syst_grid = f(time, wl)
    
    plt.figure('syst_grid')
    plt.plot(time, syst_grid[100])
    
    # print (np.diff(wl))
    # print (np.diff(syst_wl))
    
    plt.figure('syst_lc_ex')
    plt.plot(time, opt.lc[int(len(wl)/2)], 'r-')
    
    opt.lc *= syst_grid
    
    opt.lc_original = copy.deepcopy(opt.lc)
    
    print (opt.lc.shape)
    
    
    plt.figure('syst_lc_ex')
    plt.plot(time, opt.lc[int(len(wl)/2)], 'b-')



def run(opt):
    
    if opt.noise.ApplyRandomPRNU.val == 1:
        opt.qe = np.random.normal(1, 0.01*opt.noise.sim_prnu_rms.val, opt.fp_original[1::3,1::3].shape) # for random uncertainty
        opt.qe_uncert = np.random.normal(1, 0.01*opt.noise.sim_flat_field_uncert.val, opt.fp_original[1::3,1::3].shape) # for random uncertainty
        jexosim_msg ("RANDOM PRNU GRID SELECTED...",  opt.diagnostics)
    else:
        opt.qe = np.load('%s/data/JWST/PRNU/qe_rms.npy'%(opt.__path__))[0:opt.fp_original [1::3,1::3].shape[0],0:opt.fp_original[1::3,1::3].shape[1]]
        opt.qe_uncert = np.load('%s/data/JWST/PRNU/qe_uncert.npy'%(opt.__path__))[0:opt.fp_original[1::3,1::3].shape[0],0:opt.fp_original[1::3,1::3].shape[1]]
        jexosim_msg ("PRNU GRID SELECTED FROM FILE...", opt.diagnostics)

    opt.qe_original = copy.deepcopy(opt.qe)
    opt.qe_uncert_original = copy.deepcopy(opt.qe_uncert)

    if opt.simulation.sim_use_systematic_model.val == 1:
        add_systematic_grid(opt)
        
    return opt

      
    
