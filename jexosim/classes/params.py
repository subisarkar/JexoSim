"""
JexoSim 
2.0
Params class
v1.0

"""

import numpy as np

class Params():

    def __init__(self, opt, input_file, stage):
        
        self.opt = opt
                            
        with open(input_file) as f:
            content = f.readlines()
        # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content] 
        
        params= {}
        
        for i in range(len(content)):
             
            if content[i] != '' and content[i][0] != '#':
                aa = content[i].split()
                
                key_value = ''
                for j in range(1,len(aa)):
                    key_value +=  aa[j] +' '         
                key_value = key_value[:-1]
                
                try:
                    float(key_value)
                    cond=1
                except ValueError:
                    cond=0
                if cond ==1:
                    key_value = np.float(key_value)  
                    
                params[aa[0]] = key_value # set up dictionary keys from first element in each row               

        self.params = params

        if stage == 0:
            self.stage_0()        
        elif stage == 1:
            self.stage_1()
        elif stage ==2:
            self.stage_2()
            
            

    def stage_0(self):    

        attr_dict_list = [vars(self.opt.common)]
        
        for attr_dict in attr_dict_list:
            for key in self.params.keys():
                # if 'database' in key:
                testkey = key
                # testkey = (key.replace('database_', ''))
                # print (testkey)
                for key0 in attr_dict.keys():
                    # print (key0, testkey)
                    if key0 == testkey:
                        # print (key0)
                        attr_dict2 = vars(attr_dict[key0])
                        # print (attr_dict2['val'],attr_dict2['attrib']['val'])
                        if self.params[key] !='': # if blank keeps default
                            attr_dict2['val'] = self.params[key]
                            attr_dict2['attrib']['val'] = self.params[key]
                        # print (attr_dict2['val'],attr_dict2['attrib']['val'])
                        # print ('x')
      
    def stage_1(self):    

        attr_dict_list = [vars(self.opt.simulation), vars(self.opt.observation),
                          vars(self.opt.pipeline),
                          vars(self.opt.exosystem_params), 
                          vars(self.opt.noise)]
        
        for attr_dict in attr_dict_list:
            for key in self.params.keys():
                # if 'database' in key:
                testkey = key
                # testkey = (key.replace('database_', ''))
                # print (testkey)
                for key0 in attr_dict.keys():
                    # print (key0, testkey)
                    if key0 == testkey:
                        # print (key0)
                        attr_dict2 = vars(attr_dict[key0])
                        # print (attr_dict2['val'],attr_dict2['attrib']['val']) 
                        if self.params[key] !='': # if blank keeps default
                            attr_dict2['val'] = self.params[key]
                            attr_dict2['attrib']['val'] = self.params[key]
                        # print (attr_dict2['val'],attr_dict2['attrib']['val'])
                        # print ('x')
                               
        
    def stage_2(self):   
        
        attr_dict_list = [vars(self.opt.channel.detector_readout) , vars(self.opt.channel.pipeline_params) ]
        
        for attr_dict in attr_dict_list:
            for key in self.params.keys():
                # if 'database' in key:
                testkey = key
                if testkey == 'obs_n_reset_groups':
                    testkey = 'nRST'
                # testkey = (key.replace('database_', ''))
                # print (testkey)
                for key0 in attr_dict.keys():
                    # print (key0, testkey)
                    if key0 == testkey:
                        if self.params[key] !='': # if blank keeps default  
                            cond = 1
                            if key0 == 'nRST':
                                if self.params[key] == 'default' or self.params[key] =='':
                                    cond =0
                            if cond ==1:
                                attr_dict2 = vars(attr_dict[key0])
                                # print (attr_dict2['val'],attr_dict2['attrib']['val']) 
                                attr_dict2['val'] = self.params[key]
                                attr_dict2['attrib']['val'] = self.params[key]
                                # print (attr_dict2['val'],attr_dict2['attrib']['val'])
                                # print ('x')
           
  
