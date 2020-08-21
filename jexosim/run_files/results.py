"""
Created on Fri Aug  7 08:42:22 2020
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import jexosim
from jexosim.classes.params import Params
import os

def run(results_file):
    
    jexosim_path =  os.path.dirname((os.path.dirname(jexosim.__file__)))
    paths_file ='%s/jexosim/input_files/jexosim_input_paths.txt'%(jexosim_path)
    
    params_to_opt = Params(0, paths_file, 3)  
    paths = params_to_opt.params
    

    paths = params_to_opt.params
    output_directory = paths['output_directory']
    results_file =  '%s/%s'%(output_directory, results_file)

    with open(results_file, 'rb') as handle:
       res_dict = pickle.load(handle)
        
    no_list = np.array([ 'All noise','All photon noise','Source photon noise','Dark current noise',
                        'Zodi noise','Emission noise','Read noise','Spatial jitter noise',
                        'Spectral jitter noise','Combined jitter noise','No noise - no background','No noise - all background'])  
    color = ['0.5','b', 'b','k','orange','pink', 'y','g','purple','r', '0.8','c']
              
    ch =res_dict['ch']
    
    
    if ch == 'NIRSpec_G140M_F100LP': wavlim=[0.97, 1.84]
    if ch == 'NIRSpec_G235M_F170LP': wavlim=[1.66, 3.07]
    if ch == 'NIRSpec_G395M_F290LP': wavlim=[2.87, 5.1]
    if ch == 'NIRSpec_PRISM': wavlim=[0.6, 5.3]
    if ch == 'MIRI_LRS_slitless': wavlim=[0.5, 12.0]
    if ch == 'NIRCam_F444W': wavlim=[3.9, 5.0]  
    if ch == 'NIRCam_F322W2': wavlim=[2.4, 4.0]  
    if ch == 'NIRISS_SOSS_ORDER_1': wavlim=[0.9, 2.8]
                                            
 
    for key in res_dict.keys():
        print (key)
        
    if res_dict['simulation_mode'] == 1 or res_dict['simulation_mode'] == 3:
        no_dict =  res_dict['noise_dic']  
        for key in no_dict.keys():
            print (key)
        
        for key in no_dict.keys():
            
            
            idx = np.argwhere(no_list==key)[0].item()
            col = color[idx]
            
            idx = np.argwhere ((no_dict[key]['wl']>=wavlim[0])&(no_dict[key]['wl']<=wavlim[1])).T[0]
            

            noise_type = key
            wl = no_dict[key]['wl'][idx]
            
            if res_dict['simulation_realisations'] == 1:          
                sig_stack = no_dict[key]['signal_mean_stack'][idx]
                no_stack = no_dict[key]['signal_std_stack'][idx]
                fracNoT14_stack = no_dict[key]['fracNoT14_stack'][idx]
            else:
                sig_stack = no_dict[key]['signal_mean_stack'][:,idx]
                no_stack = no_dict[key]['signal_std_stack'][:,idx]
                fracNoT14_stack = no_dict[key]['fracNoT14_stack'][:,idx]
            
            sig_mean = no_dict[key]['signal_mean_mean'][idx]
            no_mean = no_dict[key]['signal_std_mean'][idx]
            fracNoT14_mean = no_dict[key]['fracNoT14_mean'][idx]
    
            plt.figure('signal %s'%(res_dict['time_tag']))
            plt.plot(wl,sig_mean, 'o-', color = col, label = noise_type)
            if res_dict['simulation_realisations'] > 1:    
                for i in range(sig_stack.shape[0]):
                    plt.plot(wl,sig_stack[i], ':', color = col, alpha=0.5)           
            plt.legend()
            plt.grid(True)
     
            plt.figure('noise %s'%(res_dict['time_tag']))
            plt.plot(wl,no_mean, 'o-', color = col, label = noise_type)
            if res_dict['simulation_realisations'] > 1:
                for i in range(no_stack.shape[0]):
                    plt.plot(wl,no_stack[i], '.', color = col, alpha=0.5)           
            plt.legend()
            plt.grid(True)
            
            
            plt.figure('fractional noise %s'%(res_dict['time_tag']))
            plt.semilogy(wl,fracNoT14_mean, 'o-', color = col, label = noise_type)
            if res_dict['simulation_realisations'] > 1:
                for i in range(fracNoT14_stack.shape[0]):
                    plt.plot(wl,fracNoT14_stack[i], '.', color = col, alpha=0.5)           
            plt.legend()
            plt.grid(True)
            
        
    if res_dict['simulation_mode'] == 2:
        
            p_stack = res_dict['p_stack']
            p_mean = res_dict['p_mean']
            p_std = res_dict['p_std']
            wl = res_dict['wl']
            cr = res_dict['input_spec']
            cr_wl = res_dict['input_spec_wl']
       
            plt.figure('spectrum %s'%(res_dict['time_tag']))
            if res_dict['simulation_realisations'] > 1:    
                for i in range(p_stack.shape[0]):
                    plt.plot(wl,p_stack[i], '.', color='0.5', alpha =0.2)
        
            plt.plot(wl,p_mean, 'o-', color='b', label = 'mean recovered spectrum')
            plt.errorbar(wl,p_mean,p_std, ecolor='b')
            plt.plot(cr_wl,cr, ':', color='r', label='input spectrum')
            plt.legend()
            plt.grid(True)
                   

            r = 4 
            if ch== 'NIRCam_F444W' or ch== 'NIRCam_F322W2':
                r=2
            if ch== 'NIRSpec_G140M_F100LP' or ch == 'NIRSpec_G235M_F170LP' or ch == 'NIRSpec_G395M_F290LP':
                r = 4
            
            idx = np.argwhere ((res_dict['wl']>=wavlim[0])&(res_dict['wl']<=wavlim[1])).T[0]

            wav =res_dict['wl'][idx]
            p_std =res_dict['p_std'][idx]*1e6
             
            z = np.polyfit(wav, p_std, r)
                                         
            plt.figure('precision %s'%(res_dict['time_tag']))
            plt.plot(wav, p_std, 'bo', alpha=0.5)
    
            p= np.poly1d(z)
            # yhat = p(wav)
            # ybar = sum(p_std)/len(p_std)
            # SST = sum((p_std - ybar)**2)
            # SSreg = sum((yhat - ybar)**2)
            # R2 = SSreg/SST  
            y =0
            for i in range (0,r+1):
                y = y + z[i]*wav**(r-i) 
                
            plt.figure('precision %s'%(res_dict['time_tag']))
            plt.plot(wav, y, '-', color='r', linewidth=2) 
            plt.grid(True)
                   
    plt.show()
    # # plt.errordicar(res_dict['wl'],res_dict['p_mean'],res_dict['p_std'])
    # plt.errordicar(res_dict['wl'],res_dict['signal_mean'])

if __name__ == "__main__":     
    
    # run('full_transit_MIRI_LRS_slitless_GJ 1214 b_2020_08_17_1248_20.pick0le')      
    # run('Noise_budget_MIRI_LRS_slitless_GJ 1214 b_2020_08_18_2028_22.pickle')
    # run('Noise_budget_MIRI_LRS_slitless_GJ 1214 b_2020_08_18_2118_03.pickle')
    # run('Noise_budget_MIRI_LRS_slitless_GJ 1214 b_2020_08_19_1830_54.pickle')
    
    # run('Noise_budget_MIRI_LRS_slitless_GJ 1214 b_2020_08_19_1844_17.pickle')
    
    # run('Noise_budget_MIRI_LRS_slitless_GJ 1214 b_2020_08_19_1910_59.pickle')
    
    # run('Noise_budget_MIRI_LRS_slitless_GJ 1214 b_2020_08_19_1916_18.pickle')
    
    # run('Noise_budget_MIRI_LRS_slitless_GJ 1214 b_2020_08_19_1931_08.pickle')
    
    # run('Noise_budget_MIRI_LRS_slitless_GJ 1214 b_2020_08_19_1936_47.pickle')
    
    run('Full_transit_MIRI_LRS_slitless_K2-18 b_2020_08_21_1007_25.pickle')
