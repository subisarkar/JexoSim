"""
JexoSim 
2.0
Timeline module
v1.0

"""
import numpy as np
from astropy import units as u
from jexosim.lib.jexosim_lib import jexosim_msg

def run(opt):

  planet = opt.planet
  opt.T14 =   (planet.calc_T14((planet.planet.i).to(u.rad),
		 (planet.planet.a).to(u.m), 
		 (planet.planet.P).to(u.s), 
		 (planet.planet.R).to(u.m), 
		 (planet.planet.star.R).to(u.m))  ).to(u.hour)    
  opt.time_at_transit      = opt.T14*(0.5+opt.timeline.before_transit())
  
  NDR_time_estimate  = opt.t_int/(opt.channel.detector_readout.multiaccum()-1)

  jexosim_msg ("NDR time estimate", NDR_time_estimate)
  
   
  opt.frame_time = opt.channel.detector_readout.t_f.val   # Frame exposure, CLK
  opt.T14 = (opt.T14).to(u.s)
  
  opt.exposure_time   = opt.channel.detector_readout.exposure_time() # Exposure time
# Estimate NDR rates
  opt.multiaccum     = opt.effective_multiaccum    # Number of NDRs per exposure
  opt.allocated_time = (opt.channel.detector_readout.nGND()+
		      opt.channel.detector_readout.nNDR0()+
		      opt.channel.detector_readout.nRST()) * opt.frame_time
    
  jexosim_msg ('exposure_time %s'%(opt.exposure_time), opt.diagnostics)
  jexosim_msg ('allocated_time %s'%(opt.allocated_time) ,  opt.diagnostics)
  jexosim_msg ('effective multiaccum %s'%(opt.effective_multiaccum),  opt.diagnostics)
  NDR_time       = (opt.exposure_time-opt.allocated_time)/(opt.multiaccum-1)
        
     
  jexosim_msg ("initial NDR time %s"%(NDR_time ),  opt.diagnostics)
   
  jexosim_msg ("overheads %s %s %s"%(opt.channel.detector_readout.nGND()*opt.frame_time, opt.channel.detector_readout.nNDR0()*opt.frame_time, opt.channel.detector_readout.nRST()*opt.frame_time),  opt.diagnostics)
    
  # find number of frames in each non-zeroth NDR  
  nNDR = np.round(NDR_time/opt.frame_time).astype(np.int).take(0)
        
  base = [opt.channel.detector_readout.nGND.val, opt.channel.detector_readout.nNDR0.val]
  for x in range(opt.multiaccum-1): base.append(nNDR)
  base.append(opt.channel.detector_readout.nRST.val)  
    
 # Recalculate exposure time and estimates how many exposures are needed
  opt.exposure_time = sum(base)*opt.frame_time
  opt.frames_per_exposure = sum(base)
  
  
  if opt.timeline.apply_lc.val ==1:
      jexosim_msg ("Since light curve is implemented, observing time is set to 2x T14",  opt.diagnostics)
      opt.timeline.use_T14.val = 1
  
  if opt.timeline.use_T14.val ==1:
      total_observing_time = opt.T14*(1.0+opt.timeline.before_transit()+opt.timeline.after_transit())
      number_of_exposures = np.ceil((total_observing_time.to(u.s)/opt.exposure_time.to(u.s))).astype(np.int)
      number_of_exposures = number_of_exposures.value
  elif opt.timeline.use_T14.val ==0:   
      number_of_exposures = int(opt.timeline.n_exp.val)    
      if opt.timeline.obs_time.val >0:
            number_of_exposures = int(opt.timeline.obs_time.val.to(u.s) / opt.exposure_time) 
  opt.n_exp = number_of_exposures
 
  opt.total_observing_time = opt.exposure_time*opt.n_exp
  
  jexosim_msg ("number of integrations %s"%(number_of_exposures),  opt.diagnostics)
  jexosim_msg ("number of NDRs %s"%(number_of_exposures*opt.multiaccum),  opt.diagnostics)
  jexosim_msg ("total observing time (hrs) %s"%((number_of_exposures*opt.exposure_time/3600).value),  opt.diagnostics)
  jexosim_msg ("T14 %s"%(opt.T14),  opt.diagnostics)
  

  opt.frame_sequence=np.tile(base, number_of_exposures) # This is Nij
  opt.time_sequence = opt.frame_time * opt.frame_sequence.cumsum() # This is Tij
  
  # End time of each NDR
  opt.ndr_end_time = np.dstack([opt.time_sequence[1+i::len(base)] \
      for i in range(opt.multiaccum)]).flatten()
      
             
  jexosim_msg ("actual NDR time %s"%(opt.ndr_end_time[1]-opt.ndr_end_time[0]  ),  opt.diagnostics) 
  jexosim_msg ("final exposure time %s"%(opt.exposure_time ) , opt.diagnostics) 
     
 
    # Number of frames contributing to each NDR
  opt.frames_per_ndr = np.dstack([opt.frame_sequence[1+i::len(base)] \
      for i in range(opt.multiaccum)]).flatten()
           
    # CLK counter of each NDR
  opt.ndr_end_frame_number = np.round(opt.ndr_end_time/opt.frame_time).astype(int).value
  opt.duration_per_ndr  = opt.frames_per_ndr*opt.frame_time 
  opt.n_ndr = number_of_exposures*opt.multiaccum
  
 
         
  return opt

