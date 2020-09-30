"""
JexoSim
2.0 
Options class
v1.0

"""

import xml.etree.ElementTree as ET
import numpy as np
import jexosim
from astropy import units as u
import os
from   ..  import __path__
 

class Entry(object):
  val       = None
  attrib    = None
  xml_entry = None
  
  def __call__(self):
    return self.val
    
  def parse(self, xml):
    self.attrib = xml.attrib
    for attr in self.attrib.keys():
      setattr(self, attr, self.attrib[attr])     
    if hasattr(self, 'units'):
      try:
        if self.units =="":
            self.val = np.float(self.val)
        else:
            # unit = aq.unit_registry[self.units]
            # self.val = aq.Quantity(np.float64(self.val), unit)     
            self.val = u.Quantity(np.float64(self.val), self.units)  
         
 
      except (ValueError, LookupError):
          pass
        # print ('unable to convert units in entry [tag, units, value]: ', \
        #                          xml.tag, self.units, self.val)

class Options(object):    

  opt = None
  
  def __init__(self, filename = None, default_path = None):
      
    self.opt = self.parser(ET.parse(filename).getroot())
    
    jexosim_path =  os.path.dirname((os.path.dirname(jexosim.__file__)))

    if self.opt.type.name == "common configuration":
    
        if default_path:
          setattr(self.opt, "__path__", default_path)
        elif hasattr(self.opt.common, "config_path"):
          setattr(self.opt, "__path__", 
                os.path.expanduser(self.opt.common.config_path().replace('__path__', __path__[0])))        
        
        wl_delta = self.opt.common.wl_min()/ self.opt.common.logbinres()
        # print (wl_delta.value)
        # print (self.opt.common.wl_min.val.value)
        
        # print (np.arange(self.opt.common.wl_min.val.value,
        #             self.opt.common.wl_max.val.value,
        #             wl_delta.value))

        setattr(self.opt.common, 'common_wl', (np.arange(self.opt.common.wl_min.val.value,
                    self.opt.common.wl_max.val.value,
                    wl_delta.value)* wl_delta.unit))
        
        if hasattr(self.opt.common, "output_directory"):
            self.opt.common.output_directory.val = '%s/output'%(jexosim_path)
 
         
    if self.opt.type.name == "channel configuration":
        if hasattr(self.opt.channel.detector_array, "subarray_list"):
            string =  self.opt.channel.detector_array.subarray_list.val
            self.opt.channel.detector_array.subarray_list.val = list(string.split(" "))
        if hasattr(self.opt.channel.detector_array, "subarray_geometry_list"):
            string =  self.opt.channel.detector_array.subarray_geometry_list.val
            aa = list(string.split(" "))
            bb = []
            for i in range(len(aa)):
                idx = aa[i].find(',')
                x = np.int(aa[i][:idx]); y =np.int(aa[i][idx+1:])
                bb.append([x,y])
            self.opt.channel.detector_array.subarray_geometry_list.val = bb
        if hasattr(self.opt.channel.detector_array, "subarray_t_f_list"):
            print (self.opt.channel.detector_array.subarray_t_f_list.val)
            string =  str(self.opt.channel.detector_array.subarray_t_f_list.val)
            aa = list(string.split(" "))
            bb = []
            for i in range(len(aa)):
                if aa[i] != 's':
                    bb.append(np.float(aa[i])*u.s)
            self.opt.channel.detector_array.subarray_t_f_list.val = bb
        if hasattr(self.opt.channel.detector_readout, "pattern_list"):
            string =  self.opt.channel.detector_readout.pattern_list.val
            self.opt.channel.detector_readout.pattern_list.val = list(string.split(" "))
        if hasattr(self.opt.channel.detector_readout, "pattern_params_list"):
            string =  self.opt.channel.detector_readout.pattern_params_list.val
            aa = list(string.split(" "))
            bb = []
            for i in range(len(aa)):
                idx = aa[i].find(',')
                x = np.int(aa[i][:idx]); y =np.int(aa[i][idx+1:])
                bb.append([x,y])
            self.opt.channel.detector_readout.pattern_params_list.val = bb           
        if hasattr(self.opt.channel.detector_array, "subarray_gap_list"):
            string =  self.opt.channel.detector_array.subarray_gap_list.val
            aa = list(string.split(" "))
            for i in range(len(aa)):
                aa[i] = aa[i].split(',')
                for j in range(len(aa[i])):
                    aa[i][j] = np.float(aa[i][j])
            self.opt.channel.detector_array.subarray_gap_list.val = aa 
            


  def parser(self, root):
   
    obj = Entry()    
    for ch in root:

      retval = self.parser(ch)
      retval.parse(ch)     
                 
      if hasattr(obj, ch.tag):
        if isinstance(getattr(obj, ch.tag), list):
          getattr(obj, ch.tag).append(retval)
        else:
          setattr(obj, ch.tag,  [getattr(obj, ch.tag), retval])
      else:
        setattr(obj, ch.tag, retval)
    return obj
    



        
if __name__ == "__main__":
  opt = Options()
