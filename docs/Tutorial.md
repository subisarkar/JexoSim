
# JexoSim 2.0
<div align="center">
<img src="jexosim_logo.png" width="300px">
</img>
<br/>
</div>
<br/><br/>

# JexoSim 2.0 tutorial
 
Five example files are provided corresponding to the case studies given in Sarkar & Madhusudhan (2021) JexoSim 2.0: End-to-end JWST Simulator for Exoplanet Spectroscopy - Implementation and Case Studies (accepted).

Example 1 : OOT simulation with Allan deviation analysis (NIRISS)
------
Navigate to inside the `JexoSim` folder. 

      cd JexoSim
      
Enter the following.
      
      python run_jexosim.py jexosim_input_params_ex1.txt
      
Alternately if using an IDE (e.g. Spyder), you can open the file `JexoSim/jexosim/run_files/run_jexosim.py` and run from within the environment using the parameter file name (e.g. `jexosim_input_params_ex1.txt`) as the argument for the function `run`.  

This example will run an out-of-transit simulation, followed by Allan deviation analysis.  It will automatically run  `results.py` to display the final results.  The results can also be displayed by entering the following:
 
    python results.py xxxx.pickle

where  `xxxx.pickle`  is the output file name in the output directory.  The results will show the signal, noise, fractional noise at T14, and the predicted noise on the transit depth.

The example file is set up to run NIRISS with K2-18 b as the target.  You should get results similar to those shown below.  In order (left to right): example integration image, noiseless signal,  noise (standard deviation of signal), fractional noise at T14 vs wavelength,  precision on transit depth vs wavelength,  example spectrum (1 transit), example spectrum (10 transits), example spectrum (100 transits).

<div align="center">
<img src="example_1/case1_1.png" width="400px">
<img src="example_1/case1_2.png" width="400px">
<img src="example_1/case1_3.png" width="400px">
<img src="example_1/case1_4.png" width="400px">
<img src="example_1/case1_5.png" width="400px">
<img src="example_1/case1_6.png" width="400px">
<img src="example_1/case1_7.png" width="400px">
<img src="example_1/case1_8.png" width="400px">
</img>
<br/>
</div>
<br/><br/>


Example 2 : Full transit simulation with Monte Carlo method (NIRCam)
------
Navigate to inside the `JexoSim` folder. 

      cd JexoSim

Enter the following.

      python run_jexosim.py jexosim_input_params_ex2.txt

Alternately if using an IDE (e.g. Spyder), you can open the file `JexoSim/jexosim/run_files/run_jexosim.py` and run from within the environment using the parameter file name (e.g. `jexosim_input_params_ex2.txt`) as the argument for the function `run`.  

This example will run a Monte Carlo full transit simulation with 50 realizations.  It will automatically run  `results.py` to display the final results.  The results can also be displayed by entering the following:

    python results.py xxxx.pickle

where  `xxxx.pickle`  is the output file name in the output directory.  The results will show the the predicted noise on the transit depth and spectra with error bars.

The example file is set up to run NIRCam F322W2 with K2-18 b as the target.  You should get results similar to those shown below.  In order (left to right): example integration image, mean transit depth and distribution from the Monte Carlo, precision on transit depth vs wavelength, example spectrum (1 transit), example spectrum (1 transits), example spectrum (100 transits).
 
 <div align="center">
 <img src="example_2/case2_1.png" width="400px">
 <img src="example_2/case2_2.png" width="400px">
 <img src="example_2/case2_3.png" width="400px">
 <img src="example_2/case2_4.png" width="400px">
 <img src="example_2/case2_5.png" width="400px">
 <img src="example_2/case2_6.png" width="400px">
 </img>
 <br/>
 </div>
 <br/><br/>

You can change the `obs_inst_config` factor in the input parameters to other NIRCam configurations, or use MIRI LRS.  



Example 3 : Full transit simulation with Monte Carlo method (NIRSpec)
------
Navigate to inside the `JexoSim` folder. 

      cd JexoSim

Enter the following.

      python run_jexosim.py jexosim_input_params_ex3.txt

Alternately if using an IDE (e.g. Spyder), you can open the file `JexoSim/jexosim/run_files/run_jexosim.py` and run from within the environment using the parameter file name (e.g. `jexosim_input_params_ex3.txt`) as the argument for the function `run`.  

This example will run a Monte Carlo full transit simulation with 50 realizations.  It will automatically run  `results.py` to display the final results.  The results can also be displayed by entering the following:

    python results.py xxxx.pickle

where  `xxxx.pickle`  is the output file name in the output directory.  The results will show the the predicted noise on the transit depth and spectra with error bars.

The example file is set up to run NIRSpec G395M F290LP with the SUB2048 subarray, with K2-18 b as the target.  You should get results similar to those shown below. In order (left to right): example integration image, mean transit depth and distribution from the Monte Carlo, precision on transit depth vs wavelength, example spectrum (1 transit), example spectrum (1 transits), example spectrum (100 transits).

<div align="center">
<img src="example_3/case3_1.png" width="400px">
<img src="example_3/case3_2.png" width="400px">
<img src="example_3/case3_3.png" width="400px">
<img src="example_3/case3_4.png" width="400px">
<img src="example_3/case3_5.png" width="400px">
<img src="example_3/case3_6.png" width="400px">
</img>
<br/>
</div>
<br/><br/>

You can change the `obs_inst_config` factor in the input parameters to other NIRSpec configurations using medium or high resolution gratings, different subarray modes, or prism mode.   The target can be changed to HD 209458 b by changing the `planet_name` in the input parameters file from `K2-18 b` to `HD 209458 b`.


Example 4 : Full eclipse simulation with Monte Carlo method (MIRI)
------
Navigate to inside the `JexoSim` folder. 

      cd JexoSim

Enter the following.

      python run_jexosim.py jexosim_input_params_ex4.txt

Alternately if using an IDE (e.g. Spyder), you can open the file `JexoSim/jexosim/run_files/run_jexosim.py` and run from within the environment using the parameter file name (e.g. `jexosim_input_params_ex4.txt`) as the argument for the function `run`.  

This example will run a Monte Carlo secondary eclipse simulation with 50 realizations.  It will automatically run  `results.py` to display the final results.  The results can also be displayed by entering the following:

    python results.py xxxx.pickle

where  `xxxx.pickle`  is the output file name in the output directory.  The results will show the the predicted noise on the eclipse depth and spectra with error bars.

The example file is set up to run MIRI LRS slitless mode with HD 209458 b as the target.  You should get results similar to those shown below. In order (left to right): example integration image, mean eclipse depth and distribution from the Monte Carlo, precision on eclipse depth vs wavelength, example spectrum (1 eclipse).

<div align="center">
<img src="example_4/case4_1.png" width="400px">
<img src="example_4/case4_2.png" width="400px">
<img src="example_4/case4_3.png" width="400px">
<img src="example_4/case4_4.png" width="400px">
</img>
<br/>
</div>
<br/><br/>

You can change the `obs_inst_config` factor in the input parameters to other NIRSpec configurations using medium or high resolution gratings, different subarray modes, or prism mode.   The target can be changed to HD 209458 b by changing the `planet_name` in the input parameters file from `K2-18 b` to `HD 209458 b`.



Example 5 : Noise budget analysis (MIRI)
------
Navigate to inside the `JexoSim` folder. 

      cd JexoSim

Enter the following.

    python run_jexosim.py jexosim_input_params_ex5.txt

Alternately if using an IDE (e.g. Spyder), you can open the file `JexoSim/jexosim/run_files/run_jexosim.py` and run from within the environment using the parameter file name (e.g. `jexosim_input_params_ex3.txt`) as the argument for the function `run`.  

This example will run an out of transit simulation, cycling over all noise sources.  It will automatically run  `results.py` to display the final results.  The results can also be displayed by entering the following:

python results.py xxxx.pickle

where  `xxxx.pickle`  is the output file name in the output directory.  The results will show the the signal and noise per spectral bin, per noise source.

The example file is set up to run MIRI LRS with K2-18 b as the target.  You should get results similar to those shown below.  In order (left to right): example integration image, noiseless signal,  noise (standard deviation of signal), 
fractional noise at T14 vs wavelength.
<div align="center">
<img src="example_5/case5_1.png" width="400px">
<img src="example_5/case5_2.png" width="400px">
<img src="example_5/case5_3.png" width="400px">
<img src="example_5/case5_4.png" width="400px">

</img>
<br/>
</div>
<br/><br/>








