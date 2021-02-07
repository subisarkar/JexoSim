"""
jexosim 
 
Output module

Packages data from one observation into FITS format, with associated metadata file

"""

from jexosim.lib import output_lib

def run(opt):
    if opt.simulation.sim_output_type.val == 2:
        filename = output_lib.write_to_fits(opt)
    elif opt.simulation.sim_output_type.val == 3:
        filename = output_lib.write_to_fits_intermediate(opt)
    return filename
    
    
    

    
   


