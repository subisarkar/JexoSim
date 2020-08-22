# JexoSim

JexoSim (JWST Exoplanet Observation Simulator) is a time-domain simulator for the James Webb Space Telescope, that simulates exoplanet transit observations using all four instruments.

Installation
------
We recommend installing Anaconda, and then setting up a virtual environment for JexoSim to avoid package conflicts.  This is will also install numpy, numba, matplotlib and setuptools, which are best installed at this step for stability.  

    conda create -n jexosim python=3.8 numpy numba matplotlib setuptools

Then activate this environment. Depending on the version you have, the activation command may be any one of the following:

    source activate jexosim
    
or    

    conda activate jexosim
    
or    
    
    activate jexosim
      

### Databases

First thing to do is to download the following databases:  
[Pandeia](https://stsci.app.box.com/v/pandeia-refdata-v1p5p1/) (Pontoppidan, K.M. et al. (2016). Proc. SPIE, 9910, 991016)  
[Phoenix BT-Settl database](https://phoenix.ens-lyon.fr/Grids/BT-Settl/CIFIST2011_2015/FITS/BT-Settl_M-0.0a+0.0.tar) (Allard F., Homeier D., Freytag B., 2012, Philos. Trans. Royal Soc. A, 370, 2765)  
Unzip and fully extract the folders.

From the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=PS&constraint=default_flag=1), download the 'Planetary Systems' table as follows: under 'Download Table' choose 'CSV format', 'Download all columns', 'Download all rows', and then hit 'Download Table'.

Download the following folders:  
[PSF](https://drive.google.com/file/d/1YFbB02IR9U-9J6UDw8SsdBo5aKh0V-_6/view?usp=sharing)  
[LDC](https://drive.google.com/file/d/1lWRdqW_wI3y31ugqq2HfyyekGyOSteL_/view?usp=sharing)  
These contain pre-calculated point spread functions obtained using WebbPSF (Perrin. M. et al. (2014). Proc. SPIE. 9143, 91433X) and limb darkening coefficients obtained using ExoTETHyS (Morello, G. et al. (2020). AJ, 159,  75) .


### GitHub

Clone the repository from github, and run `setup.py`.  This will setup the remaining package dependencies for JexoSim.  It will also extract transmission files and generate wavelength solutions from the Pandeia database, producing JexoSim-compatible files, and a PRNU grid.

    git clone https://github.com/subisarkar/JexoSim.git
    cd JexoSim
    python setup.py install
    
Move the four folders and the one .csv file you downloaded in the 'Databases' step into the `JexoSim/databases/` folder.
    
    
### Output folder
By default JexoSim will place the results from the simulations into the `JexoSim/output/` folder. However you may choose a different location to store these results.  To do so find the file `jexosim_paths.txt` in the `JexoSim/jexosim/input_files/` folder, and next to `output_directory` (leaving at least one white space gap),  enter the full path to the location including folder name, e.g. `/Users/UserA/Desktop/JexoSim_Results`.  The folder will be automatically generated. If this is blank, the default location will be chosen.


Running a simulation
------
Navigate to inside the `JexoSim` folder, and run the `run_jexosim.py` file with an input parameter file (e.g. `jexosim_input_params_ex1.txt`) as the argument.  Some example input parameter files are provided (see below).

      cd JexoSim
      python run_jexosim.py jexosim_input_params_ex1.txt
      
Alternately if using an IDE (e.g. Spyder), you can open the file `JexoSim/jexosim/run_files/run_jexosim.py` and run from within the environment using the parameter file name (e.g. `jexosim_input_params_ex1.txt`) as the argument for the function `run`.  Results will be packaged as a .pickle file in the designated output folder.  The code will also display results once completed.

Results
------
The results from any given simulation are placed in the output folder.  To identify the simulation, the file name includes the type of simulation, the planet, instrument channel and end time of the simulation.  A file with the suffix 'TEMP' is a temporary file for an ongoing multi-realisation simulation, but can be accessed in the same way as a completed simulation file.  The data is in the form of a dictionary.   To display results from a results file (e.g. `Noise_budget_MIRI_LRS_slitless_GJ 1214 b_2020_08_19_2147_43.pickle`):

    cd JexoSim
    python results.py Noise_budget_MIRI_LRS_slitless_GJ 1214 b_2020_08_19_2147_43.pickle

Make sure the file is in the designated output directory for this to work.

Examples
------
In the folder `JexoSim/jexosim/input_files` there are example input parameter text files, which can be run using the 'Running a simulation' routine above.    

Citing
------

If you use JexoSim in your research, please cite:

Sarkar, S., Madhusudhan, N. Papageorgiou, A.  M (2020). JexoSim: A time domain simulator of exoplanet transit spectroscopy with JWST. Monthly Notices of the Royal Astronomical Society 491(1), pp. 378-397. (10.1093/mnras/stz2958)
