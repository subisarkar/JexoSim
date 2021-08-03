"""
JexoSim
2.0
Read and display results
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('classic')
import jexosim
from jexosim.classes.params import Params
from jexosim.classes.options import Options
from scipy import interpolate
import os

def run(results_file):
    
    jexosim_path =  os.path.dirname((os.path.dirname(jexosim.__file__)))
    paths_file ='%s/jexosim/input_files/jexosim_paths.txt'%(jexosim_path)
        
    params_to_opt = Params(0, paths_file, 3)  
    paths = params_to_opt.params
    output_directory = paths['output_directory']
    if output_directory == '':    
        common_file = 'jexosim/xml_files/JWST.xml'   
        opt_b = Options(filename='%s/%s'%(jexosim_path,common_file)).opt
        output_directory = opt_b.common.output_directory.val

    results_file =  '%s/%s'%(output_directory, results_file)
 
    with open(results_file, 'rb') as handle:
       res_dict = pickle.load(handle)
        
    no_list = np.array([ 'All noise','All photon noise','Source photon noise','Dark current noise',
                        'Zodi noise','Emission noise','Read noise','Spatial jitter noise',
                        'Spectral jitter noise','Combined jitter noise','No noise - no background','No noise - all background', 'Fano noise', 'Sunshield noise'])  
    color = ['0.5','b', 'b','k','orange','pink', 'y','g','purple','r', '0.8','c', 'c', 'brown']
     
    ch =res_dict['ch']
    
    sim_text = '%s.txt'%(results_file)
    
    # with open(sim_text) as f:
    #     content = f.readlines()
    #     content = [x.strip() for x in content]        
    #     for i in range(len(content)):       
    #         if content[i] != '' and content[i][0] != '#':
    #             aa = content[i].split()
    #             if aa[0] == 'Wavelength:':
    #                 wavlim=[np.float(aa[1]), np.float(aa[2])]
    wavlim = [5, 12]
        
    if res_dict['simulation_mode'] == 2:
        
            wl = res_dict['wl']     
            idx = np.argwhere ((res_dict['wl']>=wavlim[0])&(res_dict['wl']<=wavlim[1])).T[0]

            wl =res_dict['wl'][idx]
            if res_dict['simulation_realisations'] > 1:   
                p_stack = res_dict['p_stack'][:,idx]
            else:
                p_stack = res_dict['p_stack'][idx]
            p_mean = res_dict['p_mean'][idx]
            p_std = res_dict['p_std'][idx]
            
            
            cr = res_dict['input_spec']
            cr_wl = res_dict['input_spec_wl']
         
            idx0 = np.argwhere ((np.array(cr_wl)>=wavlim[0])&(np.array(cr_wl)<=wavlim[1])).T[0]   
            cr = cr[idx0]
            cr_wl = cr_wl[idx0]      
            
       
            plt.figure('spectrum %s'%(res_dict['time_tag']))
            if res_dict['simulation_realisations'] > 1:    
                for i in range(p_stack.shape[0]):
                    plt.plot(wl,p_stack[i], '.', color='0.5', alpha =0.2)         
        
            plt.plot(wl,p_mean, 'o-', color='b', label = 'mean recovered spectrum')
            plt.errorbar(wl,p_mean,p_std, ecolor='b')
            plt.plot(cr_wl,cr, '-', color='r', linewidth=2, label='input spectrum')
            plt.legend(loc='best')
            plt.ylabel('Contrast ratio')
            plt.xlabel('Wavelength ($\mu m$)')
            plt.grid(True)
                   
            r = 4 
            if 'NIRCam' in ch:
                r=2
            if 'NIRSpec' in ch:
                loc = ch.find('PRISM')
                if loc == -1:
                    r= 2   
                    
            z = np.polyfit(wl, p_std*1e6, r)
                                         
            plt.figure('precision %s'%(res_dict['time_tag']))
            plt.ylabel('1 sigma error on transit depth (ppm)')
            plt.xlabel('Wavelength ($\mu m$)')
            plt.plot(wl, p_std*1e6, 'bo', alpha=0.5)
    
            p= np.poly1d(z)
            # yhat = p(wav)
            # ybar = sum(p_std)/len(p_std)
            # SST = sum((p_std - ybar)**2)
            # SSreg = sum((yhat - ybar)**2)
            # R2 = SSreg/SST  
            y =0
            for i in range (0,r+1):
                y = y + z[i]*wl**(r-i) 
                
            plt.figure('precision %s'%(res_dict['time_tag']))
            plt.plot(wl, y, '-', color='r', linewidth=2) 
            plt.grid(True)
            
                
            for ntransits in [1,10,100]:
                f = interpolate.interp1d(cr_wl,cr, bounds_error=False)
                rand_spec = np.array(f(wl))
                if ntransits == 1:
                    plt.figure('sample spectrum for %s transit %s'%(ntransits, res_dict['time_tag']))  
                else:
                    plt.figure('sample spectrum for %s transits %s'%(ntransits, res_dict['time_tag']))  
                plt.plot(cr_wl,cr, '-', color='r', linewidth=2, label='input spectrum')
                for i in range(len(wl)):
                    rand_spec[i] = np.random.normal(rand_spec[i], y[i]/1e6/np.sqrt(ntransits))
                plt.plot(wl, rand_spec, 'o-', color='b', label = 'randomized spectrum')
                plt.errorbar(wl, rand_spec, y/1e6/np.sqrt(ntransits), ecolor='b')
                plt.legend(loc='best')
                plt.ylabel('Contrast ratio')
                plt.xlabel('Wavelength ($\mu m$)')
                plt.grid(True)




            plt.figure('bad pixels %s'%(res_dict['time_tag']))          
            plt.imshow(res_dict['bad_map'], interpolation='none', aspect='auto')
            ticks = np.arange(res_dict['bad_map'].shape[1])[0::int(res_dict['bad_map'].shape[1]/10)]  
            ticklabels =  np.round(res_dict['pixel wavelengths'], 2)[0::int(res_dict['bad_map'].shape[1]/10)]  
            plt.xticks(ticks=ticks, labels = ticklabels)
            plt.ylabel('Spatial pixel')
            plt.xlabel('Wavelength ($\mu m$)')
            
            
            plt.figure('example integration image %s'%(res_dict['time_tag']))          
            plt.imshow(res_dict['example_exposure_image'], interpolation='none', aspect='auto')
            ticks = np.arange(res_dict['example_exposure_image'].shape[1])[0::int(res_dict['example_exposure_image'].shape[1]/10)]  
            ticklabels =  np.round(res_dict['pixel wavelengths'], 2)[0::int(res_dict['example_exposure_image'].shape[1]/10)]  
            plt.xticks(ticks=ticks, labels = ticklabels)
            plt.ylabel('Spatial pixel')
            plt.xlabel('Wavelength ($\mu m$)')           
            cbar = plt.colorbar() 
            cbar.set_label('Count (e$^-$)',size=12)
            plt.xlabel('Wavelength ($\mu m$)')
            
            if 'realization_0_binned_lc' in res_dict.keys():
                plt.figure('example light curve  from first realisation')          
                lc = res_dict['realization_0_binned_lc'][:, int(res_dict['realization_0_binned_lc'].shape[1]/2)]
                wav = res_dict['wl'][int(res_dict['realization_0_binned_lc'].shape[1]/2)]
                time = res_dict['exp_end_time'] 
                plt.plot(time, lc, label = f'{np.round(wav,2)} microns')
                plt.legend(loc='best')
                plt.grid()
                plt.xlabel('Time (sec)')
                plt.ylabel('Signal (e$^-$)')   
                            
     
                
    elif res_dict['simulation_mode'] == 1:
        no_dict =  res_dict['noise_dic']  
  
        for key in no_dict.keys():
   
            idx = np.argwhere(no_list==key)[0].item()
            col = color[idx]
            
            idx = np.argwhere ((no_dict[key]['wl']>=wavlim[0])&(no_dict[key]['wl']<=wavlim[1])).T[0]          

            noise_type = key
            wl = no_dict[key]['wl'][idx]
            
            print (wl/np.gradient(wl))
            
            if res_dict['simulation_realisations'] == 1:          
                sig_stack = no_dict[key]['signal_mean_stack'][idx]
                no_stack = no_dict[key]['signal_std_stack'][idx]
                if 'fracNoT14_mean' in no_dict[key].keys():
                    fracNoT14_stack = no_dict[key]['fracNoT14_stack'][idx]
            else:
                sig_stack = no_dict[key]['signal_mean_stack'][:,idx]
                no_stack = no_dict[key]['signal_std_stack'][:,idx]
                if 'fracNoT14_mean' in no_dict[key].keys():
                    fracNoT14_stack = no_dict[key]['fracNoT14_stack'][:,idx]
            
            sig_mean = no_dict[key]['signal_mean_mean'][idx]
            no_mean = no_dict[key]['signal_std_mean'][idx]
            if 'fracNoT14_mean' in no_dict[key].keys():
                fracNoT14_mean = no_dict[key]['fracNoT14_mean'][idx]
    
            plt.figure('signal %s'%(res_dict['time_tag']))
            plt.plot(wl,sig_mean, 'o-', color = col, label = noise_type)
            if res_dict['simulation_realisations'] > 1:    
                for i in range(sig_stack.shape[0]):
                    plt.plot(wl,sig_stack[i], ':', color = col, alpha=0.5)           
            plt.legend(loc='best', ncol = 3, borderpad =0.3, fontsize=10)
            plt.ylabel('Signal (e$^-$)')
            plt.xlabel('Wavelength ($\mu m$)')
            plt.grid(True)
     
            plt.figure('noise %s'%(res_dict['time_tag']))
            plt.plot(wl,no_mean, 'o-', color = col, label = noise_type)
            if res_dict['simulation_realisations'] > 1:
                for i in range(no_stack.shape[0]):
                    plt.plot(wl,no_stack[i], '.', color = col, alpha=0.5)           
            plt.legend(loc='best', ncol = 3, borderpad =0.3, fontsize=10)
            plt.ylabel('Noise (e$^-$)')
            plt.xlabel('Wavelength ($\mu m$)')
            plt.grid(True)
            
            if res_dict['simulation_realisations'] > 1:
                for i in range(no_stack.shape[0]):
                    plt.plot(wl,no_stack[i], '.', color = col, alpha=0.5)                 
            
            if 'fracNoT14_mean' in no_dict[key].keys():
                plt.figure('fractional noise %s'%(res_dict['time_tag']))
                plt.plot(wl,fracNoT14_mean, 'o-', color = col, label = noise_type)
                if res_dict['simulation_realisations'] > 1:
                    for i in range(fracNoT14_stack.shape[0]):
                        plt.plot(wl,fracNoT14_stack[i], '.', color = col, alpha=0.5)           
                plt.legend(loc='best', ncol = 3, borderpad =0.3, fontsize=10)
                plt.ylabel('Fractional noise at T14 (ppm)')
                plt.xlabel('Wavelength ($\mu m$)')
                plt.grid(True)
                plt.ylim(fracNoT14_mean.min() - fracNoT14_mean.min()*0.2
                         , fracNoT14_mean.max() + fracNoT14_mean.max()*0.2)
             
                            
                plt.figure('precision %s'%(res_dict['time_tag']))
                plt.plot(wl,fracNoT14_mean*np.sqrt(2), 'o', color = col, label = noise_type, alpha=0.5)
                plt.legend(loc='best', ncol = 3, borderpad =0.3, fontsize=10)
                plt.ylabel('1$\sigma$ error on transit depth (ppm)')
                plt.xlabel('Wavelength ($\mu m$)')
                plt.ylim(fracNoT14_mean.min()*np.sqrt(2) - fracNoT14_mean.min()*np.sqrt(2)*0.2
                         , fracNoT14_mean.max()*np.sqrt(2) + fracNoT14_mean.max()*np.sqrt(2)*0.2)
    
                r = 4 
                if 'NIRCam' in ch:
                    r=2
                if 'NIRSpec' in ch:
                    loc = ch.find('PRISM')
                    if loc == -1:
                        r= 2  
                    
            
                z = np.polyfit(wl, fracNoT14_mean*np.sqrt(2), r)
                p= np.poly1d(z)
                # yhat = p(wav)
                # ybar = sum(p_std)/len(p_std)
                # SST = sum((p_std - ybar)**2)
                # SSreg = sum((yhat - ybar)**2)
                # R2 = SSreg/SST  
                y =0
                for i in range (0,r+1):
                    y = y + z[i]*wl**(r-i) 
                
                plt.plot(wl, y, '-', color='r', linewidth=2) 
                plt.grid(True)
                
                # print (wl/np.gradient(wl))
                # aa = np.vstack((wl, y, fracNoT14_mean*np.sqrt(2))).T
                # np.savetxt('/Users/user1/Desktop/Case1_unbinned.txt', aa)
                # print (aa)
                # xxx
                
                         
                cr = res_dict['input_spec']
                cr_wl = res_dict['input_spec_wl']
 
                idx0 = np.argwhere ((np.array(cr_wl)>=wavlim[0])&(np.array(cr_wl)<=wavlim[1])).T[0]   
                cr = cr[idx0]
                cr_wl = cr_wl[idx0]
  
                f = interpolate.interp1d(cr_wl,cr, bounds_error=False)
                
                for ntransits in [1,10,100]:
                    rand_spec = np.array(f(wl))
                    plt.figure('sample spectrum for %s transit %s'%(ntransits, res_dict['time_tag']))  
                    plt.plot(cr_wl,cr, '-', color='r', linewidth=2, label='input spectrum')
                    for i in range(len(wl)):
                        rand_spec[i] = np.random.normal(rand_spec[i], y[i]/1e6/np.sqrt(ntransits))
                    plt.plot(wl, rand_spec, 'o-', color='b', label = 'randomized spectrum')
                    plt.errorbar(wl, rand_spec, y/1e6/np.sqrt(ntransits), ecolor='b')
                    plt.legend(loc='best')
                    plt.ylabel('Contrast ratio')
                    plt.xlabel('Wavelength ($\mu m$)')
                    plt.grid(True)
                
       

            plt.figure('bad pixels %s'%(res_dict['time_tag']))          
            plt.imshow(no_dict[key]['bad_map'], interpolation='none', aspect='auto')
            ticks = np.arange(no_dict[key]['bad_map'].shape[1])[0::int(no_dict[key]['bad_map'].shape[1]/10)]  
            ticklabels =  np.round(no_dict[key]['pixel wavelengths'], 2)[0::int(no_dict[key]['bad_map'].shape[1]/10)]  
            plt.xticks(ticks=ticks, labels = ticklabels)
            plt.ylabel('Spatial pixel')
            plt.xlabel('Wavelength ($\mu m$)')
            
            plt.figure('example integration image %s'%(res_dict['time_tag']))          
            plt.imshow(no_dict[key]['example_exposure_image'], interpolation='none', aspect='auto', vmin=0, vmax=no_dict[key]['example_exposure_image'].max(), cmap='jet')
            ticks = np.arange(no_dict[key]['example_exposure_image'].shape[1])[0::int(no_dict[key]['example_exposure_image'].shape[1]/10)]  
            ticklabels =  np.round(no_dict[key]['pixel wavelengths'], 2)[0::int(no_dict[key]['example_exposure_image'].shape[1]/10)]  
            plt.xticks(ticks=ticks, labels = ticklabels)
            plt.ylabel('Spatial pixel')
            plt.xlabel('Wavelength ($\mu m$)')           
            cbar = plt.colorbar() 
            cbar.set_label('Count (e$^-$)',size=12)


    elif res_dict['simulation_mode'] == 3 or res_dict['simulation_mode'] == 4:
        no_dict =  res_dict['noise_dic']  
  
        for key in no_dict.keys():
            
            print (key)
   
            idx = np.argwhere(no_list==key)[0].item()
            col = color[idx]
            
            idx = np.argwhere ((no_dict[key]['wl']>=wavlim[0])&(no_dict[key]['wl']<=wavlim[1])).T[0]          

            noise_type = key
            wl = no_dict[key]['wl'][idx]
            
            if res_dict['simulation_realisations'] == 1:          
                sig_stack = no_dict[key]['signal_mean_stack'][idx]
                no_stack = no_dict[key]['signal_std_stack'][idx]
                if 'fracNoT14_mean' in no_dict[key].keys():
                    fracNoT14_stack = no_dict[key]['fracNoT14_stack'][idx]
            else:
                sig_stack = no_dict[key]['signal_mean_stack'][:,idx]
                no_stack = no_dict[key]['signal_std_stack'][:,idx]
                if 'fracNoT14_mean' in no_dict[key].keys():
                    fracNoT14_stack = no_dict[key]['fracNoT14_stack'][:,idx]
            
            sig_mean = no_dict[key]['signal_mean_mean'][idx]
            no_mean = no_dict[key]['signal_std_mean'][idx]
            if 'fracNoT14_mean' in no_dict[key].keys():
                fracNoT14_mean = no_dict[key]['fracNoT14_mean'][idx]
    
            plt.figure('signal %s'%(res_dict['time_tag']))
            plt.plot(wl,sig_mean, 'o-', color = col, label = noise_type)
            if res_dict['simulation_realisations'] > 1:    
                for i in range(sig_stack.shape[0]):
                    plt.plot(wl,sig_stack[i], ':', color = col, alpha=0.5)           
            plt.legend(loc='best', ncol = 2, borderpad =0.3, fontsize=10)
            plt.ylabel('Signal (e$^-$)')
            plt.xlabel('Wavelength ($\mu m$)')
            plt.grid(True)
     
            plt.figure('noise %s'%(res_dict['time_tag']))
            plt.plot(wl,no_mean, 'o-', color = col, label = noise_type)
            if res_dict['simulation_realisations'] > 1:
                for i in range(no_stack.shape[0]):
                    plt.semilogy(wl,no_stack[i], '.', color = col, alpha=0.5)           
            plt.legend(loc='best', ncol = 3, borderpad =0.3, fontsize=10)
            plt.ylabel('Noise (e$^-$)')
            plt.xlabel('Wavelength ($\mu m$)')
            plt.grid(True)
            
            
            # plt.figure('sigp')
            # N = 2367
            # sigp = np.sqrt(2)*(no_mean/sig_mean)/np.sqrt(N)
            # plt.plot(wl, sigp*1e6)
            
            # xxx
            
            
            
            if res_dict['simulation_realisations'] > 1:
                for i in range(no_stack.shape[0]):
                    plt.plot(wl,no_stack[i], '.', color = col, alpha=0.5)                 
            
            if 'fracNoT14_mean' in no_dict[key].keys():
                plt.figure('fractional noise %s'%(res_dict['time_tag']))
                plt.semilogy(wl,fracNoT14_mean, 'o-', color = col, label = noise_type)
                if res_dict['simulation_realisations'] > 1:
                    for i in range(fracNoT14_stack.shape[0]):
                        plt.plot(wl,fracNoT14_stack[i], '.', color = col, alpha=0.5)           
                plt.legend(loc='best', ncol = 3, borderpad =0.3, fontsize=10)
                plt.ylabel('Fractional noise at T14 (ppm)')
                plt.xlabel('Wavelength ($\mu m$)')
                plt.grid(True)
      

            if key == 'All noise':
                plt.figure('bad pixels %s'%(res_dict['time_tag']))          
                plt.imshow(no_dict[key]['bad_map'], interpolation='none', aspect='auto')
                ticks = np.arange(no_dict[key]['bad_map'].shape[1])[0::int(no_dict[key]['bad_map'].shape[1]/10)]  
                ticklabels =  np.round(no_dict[key]['pixel wavelengths'], 2)[0::int(no_dict[key]['bad_map'].shape[1]/10)]  
                plt.xticks(ticks=ticks, labels = ticklabels)
                plt.ylabel('Spatial pixel')
                plt.xlabel('Wavelength ($\mu m$)')
                
                plt.figure('example integration image %s'%(res_dict['time_tag']))          
                plt.imshow(no_dict[key]['example_exposure_image'], interpolation='none', aspect='auto', vmin=0, vmax=no_dict[key]['example_exposure_image'].max(), cmap='jet')
                ticks = np.arange(no_dict[key]['example_exposure_image'].shape[1])[0::int(no_dict[key]['example_exposure_image'].shape[1]/10)]  
                ticklabels =  np.round(no_dict[key]['pixel wavelengths'], 2)[0::int(no_dict[key]['example_exposure_image'].shape[1]/10)]  
                plt.xticks(ticks=ticks, labels = ticklabels)
                plt.ylabel('Spatial pixel')
                plt.xlabel('Wavelength ($\mu m$)')           
                cbar = plt.colorbar() 
                cbar.set_label('Count (e$^-$)',size=12)

            
    plt.show()

if __name__ == "__main__":     

    # run('OOT_SNR_MIRI_LRS_slitless_SLITLESSPRISM_FAST_K2-18_b_2020_12_28_1157_21.pickle')
    
    # run('OOT_SNR_MIRI_LRS_slitless_SLITLESSPRISM_FAST_K2-18_b_2021_02_04_1304_53.pickle')
    
    # run('Full_transit_NIRCam_TSGRISM_F322W2_SUBGRISM64_4_output_RAPID_K2-18_b_2021_02_05_1008_14.pickle')
    # run('Full_transit_NIRSpec_BOTS_G140M_F100LP_SUB2048_NRSRAPID_K2-18_b_2021_02_05_2248_21.pickle')
    
    # run('Noise_budget_MIRI_LRS_slitless_SLITLESSPRISM_FAST_K2-18_b_2021_02_07_1010_08.pickle')
    
    # run('OOT_SNR_NIRISS_SOSS_GR700XD_SUBSTRIP96_NISRAPID_K2-18_b_2021_02_03_1146_44.pickle')
    
    
    # run('Noise_budget_MIRI_LRS_slitless_SLITLESSPRISM_FAST_K2-18_b_2021_05_22_1333_24.pickle')
    
    run('OOT_SNR_MIRI_LRS_slitless_SLITLESSPRISM_FAST_xxx_2021_07_23_1400_07.pickle')