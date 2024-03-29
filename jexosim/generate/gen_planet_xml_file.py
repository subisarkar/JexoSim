"""
JexoSim 
2.0
Generate planet files
v1.0

"""

import numpy as np
import pandas as pd
from lxml import etree as ET # USE lxml because it maintins order when setting attributes.
from jexosim.lib import jexosim_lib, exosystem_lib
import jexosim
from astropy import units as u
import os, csv

"""
Required fields from NASA Explanet Catalogue Planetary Systems table

pl_name:        Planet Name															
hostname:       Host Name																									
pl_orbper:      Orbital Period [days]																																			
pl_orbsmax:     Orbit Semi-Major Axis [au])																	
pl_radj:        Planet Radius [Jupiter Radius]																																	
pl_bmassj:      Planet Mass or Mass*sin(i) [Jupiter Mass]																																
pl_orbeccen:    Eccentricity																															
pl_eqt:         Equilibrium Temperature [K]																
pl_orbincl:     Inclination [deg]		
pl_trandur:     Transit Duration [hours]																								
st_teff:        Stellar Effective Temperature [K]						
st_rad:         Stellar Radius [Solar Radius]									
st_mass:        Stellar Mass [Solar mass]	
st_met:         Stellar Metallicity [dex]									
elat:           Ecliptic Latitude [deg]										
sy_dist:        Distance [pc]												
sy_jmag:        J (2MASS) Magnitude		
sy_kmag:        K (2MASS) Magnitude											
pl_pubdate:     Planetary Parameter Reference Publication Date	
"""	 																																																		

def make_planet_xml_file(opt, pl):

    jexosim_path =  os.path.dirname((os.path.dirname(jexosim.__file__)))
    databases_dir = '%s/archive'%(jexosim_path)
    cond=0
    for root, dirs, files in os.walk(databases_dir):
        for dirc in files:
            if 'PS_' in dirc:
                dirc_name = dirc
                cond=1
                break
    if cond==0:
        print ('Error: database not found')    
    planet_db_path = '%s/%s'%(databases_dir, dirc_name)
       
    file=open( planet_db_path, "r")
    reader = csv.reader(file)
    ct=-1
    for line in reader:
        ct+=1
        if 'pl_name' in line:
            start_row = ct
            break
     
    data = pd.read_csv(planet_db_path, skiprows=start_row)
    
    jexosim_path =  os.path.dirname((os.path.dirname(jexosim.__file__)))

    target_folder = '%s/jexosim/xml_files/exosystems'%(jexosim_path)
    template_folder = '%s/jexosim/data/templates'%(jexosim_path)
    
    # pl = 'HD 209458 b'
    #pl = '55 Cnc e'
    #pl = 'K2-18 b'
    #pl = 'GJ 1214 b'
    # pl = 'TRAPPIST-1 d'
    #pl = 'LHS 1140 b'
    #pl = 'HD 21749 c'
    #pl = 'HD 15337 b'
    # pl = 'GJ 357 b'
    #pl = 'LTT 1445 A b'
    # pl_list = [ 'L 98-59 d', 'L 168-9 b' , 'LTT 1445 A b',  'GJ 357 b', 'HD 15337 b' , 'HD 21749 c',
    #       'LHS 1140 b', 'TRAPPIST-1 d', 'GJ 1214 b', 'K2-18 b', '55 Cnc e','HD 209458 b' ]
    
    
    # pl_list = [ 'TRAPPIST-1 d' ]
    
    pl_list=[pl]
     
    for pl in pl_list:
        
        check_xml_file = '%s/%s.xml'%(target_folder, pl)
        cond = 1
        if os.path.exists(check_xml_file ):
            print ('A file already exists for this planet')
            # if opt.exosystem_params.planet_file_renew.val == 1:
            #     cond = 1
            # else:
            #     cond = 0
            cond=0  # if need a new file, just delete the old one
        
        if cond ==1:
            print ('Making new file')
                             
            idx_list = []
            for i in range (len(data['pl_name'])):
                if data['pl_name'][i] == pl:
                    # print (i, pl)
                    idx_list.append(i)        
                    
            params = 16
            param_array = np.zeros(( len(idx_list), params))
            nan_list =[]
            ct = -1
            for idx in idx_list:
                ct+=1
                ecliptic_lat = data['elat'][idx]
                R_p = data['pl_radj'][idx]
                M_p = data['pl_bmassj'][idx]
                T_p = data['pl_eqt'][idx]
                i = data['pl_orbincl'][idx]
                e = data['pl_orbeccen'][idx]
                a = data['pl_orbsmax'][idx]
                P = data['pl_orbper'][idx]
                R_s = data['st_rad'][idx] 
                M_s = data['st_mass'][idx] 
                T_s = data['st_teff'][idx] 
                d = data['sy_dist'][idx]
                Z = data['st_met'][idx]
                T14 = data['pl_trandur'][idx]
                ra = data['ra'][idx]
                dec = data['dec'][idx]
                param_list = np.array([ecliptic_lat, R_p, M_p, T_p, i, e, a, P, R_s, M_s, T_s, d, Z, T14, ra, dec])
                param_array[ct] = param_list
                nan_list.append(len(np.argwhere(np.isnan(param_list))))
             
            # find indexes of studies where fewest params are missing    
            study_list = np.argwhere(nan_list == np.min(nan_list)).T[0] + idx_list[0] # add this at end to keep consistent with indexing in data frame
            
            # if more than one then pick the study that is most recent
            if len(study_list) >1:
                pub_date=[]
                for i in range (len(study_list)):
                    date_format  = data['pl_pubdate'][study_list[i]]
                    aa = date_format.find('-')   
                    date_new = np.float(date_format[:aa]) + np.float(date_format[aa+1:])/12.
                    pub_date.append([date_new][0])
                # index of most recent study
                idx = study_list[np.argwhere(pub_date == np.max(pub_date))[0].item()]  
            else:
                idx = study_list[0]
                
            # print ("Most recent study with most complete paramaters")
            # print (data['pl_pubdate'][idx],data['pl_refname'][idx])   
            

            
            star_name = data['hostname'][idx]
            J_mag = data['sy_jmag'][idx]
            K_mag = data['sy_kmag'][idx]
            

            # mag_band = 'J'
                
            
            # now find missing params
            
            ecliptic_lat = data['elat'][idx]
            R_p = data['pl_radj'][idx]
            M_p = data['pl_bmassj'][idx]
            T_p = data['pl_eqt'][idx]
            i = data['pl_orbincl'][idx]
            e = data['pl_orbeccen'][idx]
            a = data['pl_orbsmax'][idx]
            P = data['pl_orbper'][idx]
            R_s = data['st_rad'][idx] 
            M_s = data['st_mass'][idx] 
            T_s = data['st_teff'][idx] 
            d = data['sy_dist'][idx]
            Z = data['st_met'][idx]
            T14 = data['pl_trandur'][idx]
            ra = data['ra'][idx]
            dec = data['dec'][idx]
            param_list = np.array([ecliptic_lat, R_p, M_p, T_p, i, e, a, P, R_s, M_s, T_s, d, Z, T14, ra, dec])
            
            dic = {'elat': ecliptic_lat,
                   'pl_radj': R_p,'pl_bmassj': M_p, 'pl_eqt':T_p,  'pl_orbincl': i, 'pl_orbeccen': e, 'pl_orbsmax': a,
                   'pl_orbper': P ,'st_rad': R_s,'st_mass': M_s,'st_teff': T_s,'sy_dist': d,'st_met': Z, 'pl_trandur': T14,
                   'ra': ra, 'dec': dec}
            
            params = ['elat', 'pl_radj','pl_bmassj', 'pl_eqt', 'pl_orbincl','pl_orbeccen','pl_orbsmax',
                        'pl_orbper','st_rad','st_mass','st_teff','sy_dist','st_met', 'pl_trandur', 'ra', 'dec']
             
            
            print (param_list)
           
            missing_params = np.argwhere(np.isnan(param_list)).T[0]
            
            print ("missing parameters")
            if len(missing_params)==0:
                print ('None')
            else:
                for i in range(len(missing_params)):
                    print (params[missing_params[i]])
                          
            if len(missing_params)>0:
                print ("Now find most recent studies for missing params")
               
            data0 = data[idx_list[0]: idx_list[-1]+1]
            for i in range(len(missing_params)):
                mp = np.array( data0[params[missing_params[i]]])
                idx = np.argwhere(np.isnan(mp)) 
                mp[idx] = 1e30
                has_param = np.argwhere(mp<1e30).T[0] + idx_list[0]
                
                if len(has_param) == 0:
                    pass
                    print ("no data found for parameter: enter manually")
      
                elif len(has_param) >= 1:
                    pub_date=[]
                    for j in range (len(has_param)):
                        date_format  = data['pl_pubdate'][has_param[j]]
                        # print (date_format)
                        date_format = date_format.split() # deals with cases that have time added as 00:00
                        for possible_date in date_format:
                            if '-' in possible_date:               
                                date = possible_date
                                break
                        date_components = date.split('-')
                        # print (date_components)
                        if len(date_components) == 2:
                            date_components.append('0')
                        days_per_month = np.array([31,28,31,30,31,30,31,31,30,31,30,31])
                        days = np.sum(days_per_month[0:int(np.float(date_components[1])-1) ]) + np.float(date_components[2])                                      
                        date_new = np.float(date_components[0]) +  days/365.25
                        pub_date.append(date_new)
                    # index of most recent study
                    idx = has_param[np.argwhere(pub_date == np.max(pub_date))[0].item()]
                    # print (data['pl_pubdate'][idx],data['pl_refname'][idx])
                    dic[params[missing_params[i]]]   = data0[params[missing_params[i]]][idx]
                    param_list[missing_params[i]] = data0[params[missing_params[i]]][idx]
   
            # print (dic)

            missing_params = np.argwhere(np.isnan(param_list))
            if len(missing_params) >0:
                missing_params = np.argwhere(np.isnan(param_list)).T[0]
            
                for i in range(len(missing_params)):         
                    if  params[missing_params[i]] == 'pl_trandur' or params[missing_params[i]] == 'pl_orbeccen' or params[missing_params[i]] =='pl_eqt' :
                        pass # these can be calculated given a complete set of inputs
                    else: 
                        print (params[missing_params[i]], "still missing information: enter manually")
                        value = input("Enter value (check correct units in template xml file)")
                        dic[params[missing_params[i]]] = value
                        print (params[missing_params[i]], 'set to', value)              
                
            # calculate logg and add to dic 
            print ("Adding calculated log g")
            logg = jexosim_lib.calc_logg(M_s*u.Msun,R_s*u.Rsun)[1]
            dic['logg'] =logg
            
            # # add mag and star name to dic
            dic['hostname']= star_name
            dic['sy_jmag']= J_mag
            dic['sy_kmag']= K_mag
            # dic['mag_band']= mag_band
            
            idx = np.argwhere(np.isnan(dic['pl_orbeccen']))
            if len(idx) ==1:
                print ("As no e value found, setting e to 0")
                dic['pl_orbeccen'] = 0
                param_list[5] = 0
                         
            idx = np.argwhere(np.isnan(dic['pl_eqt']))
            if len(idx) ==1:
                T_p = jexosim_lib.calc_EqT(T_s, R_s*u.Rsun, a*u.au, 0.3, 0)
                print ("As no T_p value found, calculating T_p = ", T_p)
                dic['pl_eqt'] = T_p
                param_list[3] = T_p
                
            idx = np.argwhere(np.isnan(dic['pl_trandur']))
            if len(idx) ==1:
                pl_inc = (dic['pl_orbincl']*u.deg).to(u.rad)
                pl_a  =(dic['pl_orbsmax']*u.au).to(u.m)
                pl_P  =(dic['pl_orbper']*u.day).to(u.hr)
                pl_Rp = (dic['pl_radj']*u.Rjup).to(u.m)
                pl_Rs = (dic['st_rad']*u.Rsun).to(u.m)         
                T14 = exosystem_lib.calc_T14(pl_inc, pl_a, pl_P, pl_Rp, pl_Rs)
                print ("As no T14 value found, calculating T14 = ", T14)
                T14 = T14.value
                dic['pl_trandur'] = T14

            def makeXmlFile(xmlTemplateFile, dic, pl):
  
              tree = ET.parse(xmlTemplateFile)
             
              root = tree.getroot()
        
              for child in root.findall('model_exosystem'):
                  child.find('R_p').set('val', '%s'%(dic['pl_radj']))
                  child.find('M_p').set('val', '%s'%(dic['pl_bmassj']))
                  child.find('T_p').set('val', '%s'%(dic['pl_eqt']))
                  child.find('i').set('val', '%s'%(dic['pl_orbincl']))
                  child.find('e').set('val', '%s'%(dic['pl_orbeccen']))
                  child.find('a').set('val', '%s'%(dic['pl_orbsmax']))
                  child.find('P').set('val', '%s'%(dic['pl_orbper']))
                  child.find('albedo').set('val', '0.3')
                  child.find('R_s').set('val', '%s'%(dic['st_rad']))
                  child.find('M_s').set('val', '%s'%(dic['st_mass']))
                  child.find('T_s').set('val', '%s'%(dic['st_teff']))
                  child.find('d').set('val', '%s'%(dic['sy_dist']))
                  child.find('logg').set('val', '%s'%(dic['logg']))
                  child.find('Z').set('val', '%s'%(dic['st_met']))   
                  child.find('T14').set('val', '%s'%(dic['pl_trandur']))   
                  child.find('star_name').set('val', '%s'%(dic['hostname']))
                  child.find('ecliptic_lat').set('val', '%s'%(dic['elat']))
                  child.find('planet_name').set('val', '%s'%(pl))  
                  child.find('J_mag').set('val', '%s'%(dic['sy_jmag'])) 
                  child.find('K_mag').set('val', '%s'%(dic['sy_kmag'])) 
                  child.find('ra').set('val', '%s'%(dic['ra'])) 
                  child.find('dec').set('val', '%s'%(dic['dec'])) 
               
              newXmlFileName = '%s/%s.xml'%(target_folder, pl)
              f = open(newXmlFileName, 'wb')
              tree.write(f)
              f.close()  
                     
            xml_file = '%s/template.xml'%(template_folder)
            makeXmlFile(xml_file, dic, pl)
            
      
                
                
