# JexoSim 2.0 tutorial
 
 Four example files are provided corresponding to the case studies given in Sarkar & Madhusudhan (2021) JexoSim 2.0: Enhanced JWST Simulator for Exoplanet Spectroscopy - Implementation and Case Studies (submitted).

Example 1 : Allan deviation analysis
------
Navigate to inside the `JexoSim` folder. 

      cd JexoSim
      
Enter the following.
      
      python run_jexosim.py jexosim_input_params_ex1.txt
      
Alternately if using an IDE (e.g. Spyder), you can open the file `JexoSim/jexosim/run_files/run_jexosim.py` and run from within the environment using the parameter file name (e.g. `jexosim_input_params_ex1.txt`) as the argument for the function `run`.  

This example will run an out-of-transit simulation, followed by Allan deviation analysis.  It will automatically run  `results.py` to display the final results.  The results can also be displayed by entering the following:
 
    python results.py xxxx.pickle

where  `xxxx.pickle`  is the output file name in the output directory.  The results will show the signal, noise, fractional noise at T14, and the predicted noise on the transit depth.

The example file is set up to run NIRISS with K2-18 b as the target.  You should get results similar to those shown below.  In order (left to right): example integration image, example spectrum (1 transit), precision on transit depth vs wavelength, fractional noise at T14 vs wavelength, noise (standard deviation of signal), noiseless signal.

<div align="center">
<img src="example_1/NIRISS_1.png" width="300px">
<img src="example_1/NIRISS_2.png" width="300px">
<img src="example_1/NIRISS_3.png" width="300px">
<img src="example_1/NIRISS_4.png" width="300px">
<img src="example_1/NIRISS_5.png" width="300px">
<img src="example_1/NIRISS_6.png" width="300px">
</img>
<br/>
</div>
<br/><br/>


Example 2 : Full transit simulation with Monte Carlo method
------
Navigate to inside the `JexoSim` folder. 

      cd JexoSim

Enter the following.

      python run_jexosim.py jexosim_input_params_ex2.txt

Alternately if using an IDE (e.g. Spyder), you can open the file `JexoSim/jexosim/run_files/run_jexosim.py` and run from within the environment using the parameter file name (e.g. `jexosim_input_params_ex2.txt`) as the argument for the function `run`.  

This example will run a Monte Carlo full transit simulation with 25 realizations.  It will automatically run  `results.py` to display the final results.  The results can also be displayed by entering the following:

python results.py xxxx.pickle

where  `xxxx.pickle`  is the output file name in the output directory.  The results will show the the predicted noise on the transit depth and spectra with error bars.



Example 3 : Noise budget
------
Navigate to inside the `JexoSim` folder. 

      cd JexoSim

Enter the following.

      python run_jexosim.py jexosim_input_params_ex3.txt

Alternately if using an IDE (e.g. Spyder), you can open the file `JexoSim/jexosim/run_files/run_jexosim.py` and run from within the environment using the parameter file name (e.g. `jexosim_input_params_ex3.txt`) as the argument for the function `run`.  

This example will run an out of transit simulation, cycling over all noise sources.  It will automatically run  `results.py` to display the final results.  The results can also be displayed by entering the following:

python results.py xxxx.pickle

where  `xxxx.pickle`  is the output file name in the output directory.  The results will show the the signal and noise per spectral bin, per noise source.

