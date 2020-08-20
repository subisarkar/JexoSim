import numpy as np
import os
import jexosim
from jexosim.activate import gen_trans_files, gen_wavelength_files, gen_PRNU_grid
# from jexosim.activate import gen_limb_darkening_coeffs, gen_psf

def run():

    jexosim_path =  os.path.dirname((os.path.dirname(jexosim.__file__)))

    input_file = '%s/jexosim/input_files/jexosim_input_paths.txt'%(jexosim_path)
    
    with open(input_file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]


    # DEFAULTS
    params= {
        'database_pandeia' :'~',  'output_directory' :'~', }

    for i in range(len(content)):
        if content[i] != '' and content[i][0] != '#':
            aa = content[i].split()
            
            
            for key in params.keys():
                if aa[0]== key:
                    params[key] = ''
                    for j in range(1,len(aa)):
                        params[key] +=  aa[j] +' '
                    params[key] = params[key][:-1]


    if not os.path.exists(params['output_directory']):
            os.makedirs(params['output_directory'])


    # ==============================================================================
    # Generate transmission files and wavelength solutions from Pandeia database and put in right folder
    # ==============================================================================
    print ("...extracting transmission files")
    gen_trans_files.run(params)

    print ("...generating and filing wavelength solutions")
    gen_wavelength_files.run(params)

    # ==============================================================================
    # Generate a PRNU grid that you can use each time: give % rms and % rms uncertainty
    # ==============================================================================
    print ("...generating a PRNU grid")
    gen_PRNU_grid.run(rms=0.03,rms_uncert=0.005)


if __name__ == "__main__":
    
    run()
