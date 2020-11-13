"""
Jexosim
2.0
Generate
v 1.0

"""

import numpy as np
import os
import jexosim
from jexosim.generate import gen_trans_files, gen_quantum_yield_files, gen_wavelength_files, gen_PRNU_grid, gen_ipc_kernel

def run():

    jexosim_path =  os.path.dirname((os.path.dirname(jexosim.__file__)))
    databases_dir = '%s/archive'%(jexosim_path)
    cond=0
    for root, dirs, files in os.walk(databases_dir):
        for dirc in dirs:
            if 'pandeia' in dirc:
                dirc_name = dirc
                cond=1
                break
    if cond==0:
        print ('Error: database not found')    
    database_path = '%s/%s'%(databases_dir, dirc_name)

    # ==============================================================================
    # Generate transmission files and wavelength solutions from Pandeia database and put in right folder
    # ==============================================================================
    print ("...extracting transmission files")
    gen_trans_files.run(database_path)
    
    print ("...generating quantum yield files")
    gen_quantum_yield_files.run(database_path)  

    print ("...generating and filing wavelength solutions")
    gen_wavelength_files.run(database_path)
    
    print ("...generating ipc kernel")
    gen_ipc_kernel.run(database_path)  
    
    # # ==============================================================================
    # # Generate a PRNU grid that you can use each time: give % rms and % rms uncertainty
    # # ==============================================================================
    print ("...generating a PRNU grid")
    gen_PRNU_grid.run(rms=0.03,rms_uncert=0.005)


if __name__ == "__main__":
    
    run()
