'''
JexoSim
2.0
Recipe 5 :
Used to loop through code upto detector module
'''

import numpy as np
import time, os
from datetime import datetime
from jexosim.modules import exosystem, telescope, channel, backgrounds, output
from jexosim.modules import detector, timeline, light_curve, systematics, noise
from jexosim.lib.jexosim_lib import jexosim_msg, jexosim_plot, write_record


class recipe_5(object):
    def __init__(self, opt):
        
        opt = self.run_JexoSimA(opt)
              
    def run_JexoSimA(self, opt):
      exosystem.run(opt) 
      telescope.run(opt) 
      channel.run(opt)  
      backgrounds.run(opt)     
      detector.run(opt)
    