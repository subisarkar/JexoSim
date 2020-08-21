# JexoSim

JexoSim (JWST Exoplanet Observation Simulator) is a time-domain simulator for the James Webb Space Telescope, that simulates exoplanet transit observations using all four instruments.

Installation
------
We recommend installing Anaconda, and then setting up a virtual environment for JexoSim to avoid package conflicts:

    conda create -n jexosim python=3

Then activate this environment.

    source activate jexosim
  

### GitHub

Clone the repository from github:

    git clone https://github.com/subisarkar/JexoSim.git
    cd JexoSim
    python setup.py install
    
### Databases

Download the following databases:  [Pandeia](https://stsci.app.box.com/v/pandeia-refdata-v1p5p1/) (Pontoppidan, K.M. et al. (2016). Proc. SPIE, 9910, 991016) and [Phoenix BT-Settl database](https://phoenix.ens-lyon.fr/Grids/BT-Settl/CIFIST2011_2015/FITS/BT-Settl_M-0.0a+0.0.tar) (Allard F., Homeier D., Freytag B., 2012, Philos. Trans. Royal Soc. A, 370, 2765).  

From the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=PS&constraint=default_flag=1), download the 'Planetary Systems' table in .csv format. The following fields must be included in the table (under the table 'Select Columns'): Planet Name, Host Name, Orbital Period [days], Orbit Semi-Major Axis [au], Planet Radius [Jupiter Radius], Planet Mass or Mass*sin(i) [Jupiter Mass], Eccentricity, Equilibrium Temperature [K], Inclination [deg], Stellar Effective Temperature [K], Stellar Radius [Solar Radius], Stellar Mass [Solar mass], Stellar Metallicity [dex], Ecliptic Latitude [deg], Distance [pc], J (2MASS) Magnitude, Planetary Parameter Reference Publication Date.

In the folder `input_files`, edit the text file `jexosim_input_paths.txt` to include the complete paths to the three databases above.

Download the following folders: [PSF](https://drive.google.com/file/d/1sh7jHOA9vTQWst9dXo5DeGDXq_ZBK8IU/view?usp=sharing) and [LDC](https://drive.google.com/file/d/1Vg10lp_Pfyrii1fg2B6QhjuyTLlYImq4/view?usp=sharing).  These contain pre-calculated point spread functions obtained using WebbPSF (Perrin. M. et al. (2014). Proc. SPIE. 9143, 91433X) and limb darkening coefficients obtained using ExoTETHyS (Morello, G. et al. (2020), Astronomical Journal, 159, 75).  Move these into the 
`data` folder (`data/PSF` and `data/LDC`).

### Activation
Before JexoSim can be run for the first time, you need to activate it.  Navigate to the `JexoSim` folder and run the `activate.py` file.  

     cd JexoSim
     python activate.py
 
 This extracts transmission files and generates wavelength solutions from the Pandeia database, producing JexoSim-compatible files.  It also generates a PRNU grid.
 

Running a simulation
------
Navigate to inside the `JexoSim` folder, and run the `run_jexosim.py` file with an input parameter file (e.g. `jexosim_input_params_ex1`) as the argument.

      cd JexoSim
      python run_jexosim.py jexosim_input_params_ex1
      
Alternately if using an IDE (e.g. Spyder), you can open the file `JexoSim/jexosim/run_files/run_jexosim.py` and run from within the environment using the parameter file name (e.g. `jexosim_input_params_ex1.txt`) as the argument for the function `run`.
Results will be packaged as a .pickle file in the designated output folder.  The code will also display results once completed.

Results
------
The results from any given simulation are placed in the output folder.  To identify the simulation, the file name includes the type of simulation, the planet, instrument channel and end time of the simulation.  A file with the suffix 'TEMP' is a temporary file for an ongoining multi-realisation simulation, but can be accessed in the same way as a completed simulation file.  The data is in the form of a dictionary.   To display results from a results file (e.g. `Noise_budget_MIRI_LRS_slitless_GJ 1214 b_2020_08_19_2147_43.pickle`):

    cd JexoSim
    python results.py Noise_budget_MIRI_LRS_slitless_GJ 1214 b_2020_08_19_2147_43.pickle


Examples
------
In the folder `input_files` there are example input parameter text files, which can be run.  

Citing
------

If you use JexoSim in your research, please cite:

Sarkar, S., Madhusudhan, N. Papageorgiou, A.  M (2020). JexoSim: A time domain simulator of exoplanet transit spectroscopy with JWST. Monthly Notices of the Royal Astronomical Society 491(1), pp. 378-397. (10.1093/mnras/stz2958)