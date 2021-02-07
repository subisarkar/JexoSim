"""
JexoSim 2.0

Output library

"""

import numpy as np
from astropy import units as u
from astropy import constants as const
from astropy.io import fits
import sys, os, glob
import pickle
from jexosim.lib import output_lib
from jexosim.lib.jexosim_lib import jexosim_msg
from datetime import datetime

           
def write_record(opt, path, lab, input_text_file):
    
    textfile = '%s/%s.txt'%(path,lab)
    
    with open(textfile, "w") as f1:
        f1.write('===== Simulation values =====')
        f1.write('\n ')
        f1.write('\nPlanet:  %s'%(opt.planet.planet.name))
        f1.write('\nChannel:  %s'%(opt.channel.name))
        if opt.simulation.sim_mode.val == 3:
            f1.write('\n\nNoise option:  noise budget')          
        else:
            f1.write('\n\nNoise option:  %s'%(opt.noise_tag))     
        f1.write('\n ')

        f1.write('\nUse saturation time?:  %s'%(opt.observation.obs_use_sat.val))
        f1.write('\nSat time (to designated fraction of full well):  %s sec'%(opt.sat_time))
        f1.write('\nt_f:  %s sec'%(opt.t_f.value))
        f1.write('\nt_g:  %s sec'%(opt.t_g.value))
        f1.write('\nt_sim:  %s sec'%(opt.t_sim.value))
        f1.write('\nsubarray  :  %s'%(opt.subarray))
        
        f1.write('\nsaturation flag  :  %s'%(opt.sat_flag))
        f1.write('\nsaturation limit  :  %s electrons'%(opt.sat_limit.value))
        f1.write('\nnumber of saturated pixels per image  :  %s'%(opt.no_sat))
        f1.write('\nzero values applied to saturated pixels  :  %s'%(opt.pipeline.pipeline_bad_corr.val))
         
        f1.write('\nt_int:  %s sec'%(opt.t_int.value.item()))
        f1.write('\nt_cycle:  %s sec'%(opt.exposure_time.value.item()))  
        f1.write('\nprojected multiaccum (n groups):  %s'%(opt.projected_multiaccum))
        f1.write('\neffective multiaccum (n groups):  %s'%(opt.effective_multiaccum))
        f1.write('\nnumber of NDRs simulated:  %s'%(opt.n_ndr) )
        f1.write('\nnumber of integration cycles:  %s'%(opt.n_exp) )          
        f1.write('\n ')
        if opt.simulation.sim_output_type.val == 1: # excludes fits files
            f1.write('\nApFactor:  %s'%(opt.pipeline.pipeline_ap_factor.val) )
            f1.write('\nAperture shape:  %s'%(opt.pipeline.pipeline_ap_shape.val) )
            f1.write('\nSpectral binning:  %s '%(opt.pipeline.pipeline_binning.val) )
            if opt.pipeline.pipeline_binning.val == 'R-bin':
                        f1.write('\nBinned R power:  %s '%(opt.pipeline.pipeline_R.val) )
            else:
                      f1.write('\nBin size (pixels):  %s '%(opt.pipeline.pipeline_bin_size.val) )
        f1.write('\nWavelength: %s %s'%(opt.channel.pipeline_params.wavrange_lo.val, opt.channel.pipeline_params.wavrange_hi.val) )
    
    with open(textfile, "a") as f1:
        f1.write('\n ')
        f1.write('\n ')
        f1.write('===== Copy of input parameters file used =====')
        f1.write('\n ')
        f1.write('\n ')
      
    with open(input_text_file) as f:
        with open(textfile, "a") as f1:
            for line in f:       
                f1.write(line)
    f1.close()
    
    
def write_record_no_pipeline(opt, path, lab, input_text_file):
    
    textfile = '%s/%s.txt'%(path,lab)
    
    with open(textfile, "w") as f1:
        f1.write('===== Simulation values =====')
        f1.write('\n ')
        f1.write('\nPlanet:  %s'%(opt.planet.planet.name))
        f1.write('\nChannel:  %s'%(opt.channel.name))
        if opt.simulation.sim_mode.val == 3:
            f1.write('\n\nNoise option:  noise budget')          
        else:
            f1.write('\n\nNoise option:  %s'%(opt.noise_tag))     
        f1.write('\n ')

        f1.write('\nUse saturation time?:  %s'%(opt.observation.obs_use_sat.val))
        f1.write('\nSat time (to designated fraction of full well):  %s sec'%(opt.sat_time))
        f1.write('\nt_f:  %s sec'%(opt.t_f.value))
        f1.write('\nt_g:  %s sec'%(opt.t_g.value))
        f1.write('\nt_sim:  %s sec'%(opt.t_sim.value))
        f1.write('\nsubarray  :  %s'%(opt.subarray))
        
        f1.write('\nsaturation limit  :  %s electrons'%(opt.sat_limit.value))
         
        f1.write('\nt_int:  %s sec'%(opt.t_int.value.item()))
        f1.write('\nt_cycle:  %s sec'%(opt.exposure_time.value.item()))  
        f1.write('\nprojected multiaccum (n groups):  %s'%(opt.projected_multiaccum))
        f1.write('\neffective multiaccum (n groups):  %s'%(opt.effective_multiaccum))
        f1.write('\nnumber of NDRs simulated:  %s'%(opt.n_ndr) )
        f1.write('\nnumber of integration cycles:  %s'%(opt.n_exp) )          
        f1.write('\n ')
    
    with open(textfile, "a") as f1:
        f1.write('\n ')
        f1.write('\n ')
        f1.write('===== Copy of input parameters file used =====')
        f1.write('\n ')
        f1.write('\n ')
      
    with open(input_text_file) as f:
        with open(textfile, "a") as f1:
            for line in f:       
                f1.write(line)
    f1.close()


def write_to_fits(opt):   
    
    jexosim_msg('Save to fits file ... ', 1)
    output_directory  = opt.common.output_directory.val

    hdu        = fits.PrimaryHDU()
    hdu.header['NEXP'] = (opt.n_exp, 'Number of exposures')
    hdu.header['MACCUM_P'] = (opt.projected_multiaccum, 'Multiaccum (projected)')
    hdu.header['MACCUM_E'] = (opt.effective_multiaccum, 'Multiaccum (effective)')
    hdu.header['TEXP'] = (opt.exposure_time.value, 'Duration of each intergration cycle [s]')
    hdu.header['PLANET'] = (opt.planet.planet.name, 'Planet name')
    hdu.header['STAR'] = (opt.planet.planet.star.name, 'Star name')
    
    hdu.header['CDELT1'] = (opt.channel.detector_pixel.plate_scale_x.val.to(u.deg).value, 'Degrees/pixel')
    hdu.header['CDELT2'] = (opt.channel.detector_pixel.plate_scale_y.val.to(u.deg).value, 'Degrees/pixel')
    hdulist = fits.HDUList(hdu)

    for i in range(opt.n_exp):
        for j in range(opt.effective_multiaccum):
            hdu = fits.ImageHDU(opt.data[..., i*opt.effective_multiaccum + j].value.astype(np.float32), name = 'NDR')
            hdu.header['EXP'] = (i, 'Exposure Number')
            hdu.header['NDR'] = (j, 'NDR Number')
            hdu.header['EXTNAME'] = ('DATA')
            hdu.header['UNITS'] = ('%s'%(opt.data.unit) )
            hdulist.append(hdu)

    col1 = fits.Column(name='Wavelength {:s}'.format(opt.x_wav_osr[1::3].unit), format='E', 
 		       array=opt.x_wav_osr[1::3].value)
    col2 = fits.Column(name='Input Spectrum', format='E', 
 		       array=opt.planet.sed.sed[1::3].value)
    cols = fits.ColDefs([col1, col2])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.name = 'INPUTS'
    hdulist.append(tbhdu)

    hdu = fits.ImageHDU(opt.pointing_timeline, name = 'POINTING_TIMELINE')
    hdulist.append(hdu)
    
    hdu = fits.ImageHDU(opt.qe_grid, name = 'PRNU')
    hdulist.append(hdu)
    
    col1 = fits.Column(name='Time {:s}'.format(opt.ndr_end_time.unit), format='E', 
 		       array=opt.ndr_end_time.value)

    cols = fits.ColDefs([col1])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.name = 'TIME'
    hdulist.append(tbhdu)
 
    #write hdulist
    lab = '%s_%s'%(opt.observation.obs_channel.val, opt.exosystem_params.planet_name.val)
    time_tag = (datetime.now().strftime('%Y_%m_%d_%H%M_%S'))
    filename = 'jexosim_%s_%s'%(lab, time_tag)
    hdulist.writeto('%s/%s.fits'%(output_directory, filename))
    
    return '%s.fits'%(filename)
    
    
def write_to_fits_intermediate(opt):   
    
    jexosim_msg('Saving binned light curves to fits file ... ', 1)
    output_directory  = opt.common.output_directory.val

    hdu        = fits.PrimaryHDU()
    hdu.header['NEXP'] = (opt.n_exp, 'Number of exposures')
    hdu.header['MACCUM_P'] = (opt.projected_multiaccum, 'Multiaccum (projected)')
    hdu.header['MACCUM_E'] = (opt.effective_multiaccum, 'Multiaccum (effective)')
    hdu.header['TEXP'] = (opt.exposure_time.value, 'Duration of each intergration cycle [s]')
    hdu.header['PLANET'] = (opt.planet.planet.name, 'Planet name')
    hdu.header['STAR'] = (opt.planet.planet.star.name, 'Star name')

    hdulist = fits.HDUList(hdu)

    hdu = fits.ImageHDU(opt.pipeline_stage_1.binnedLC.value.astype(np.float32))
    hdu.header['EXTNAME'] = ('BINNED_LIGHT_CURVES')
    hdu.header['UNITS'] = ('%s'%(opt.pipeline_stage_1.binnedLC.unit) )
    hdulist.append(hdu)

    col1 = fits.Column(name='Wavelength {:s}'.format(opt.pipeline_stage_1.binnedWav.unit), format='E', 
 		       array=opt.pipeline_stage_1.binnedWav.value)
    cols = fits.ColDefs([col1])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.name = 'WAVELENGTH'
    hdulist.append(tbhdu)

    col1 = fits.Column(name='Time {:s}'.format(opt.pipeline_stage_1.exp_end_time_grid.unit), format='E', 
 		       array=opt.pipeline_stage_1.exp_end_time_grid.value)

    cols = fits.ColDefs([col1])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.name = 'TIME'
    hdulist.append(tbhdu)
 
    #write hdulist
    lab = '%s_%s'%(opt.observation.obs_channel.val, opt.exosystem_params.planet_name.val)
    time_tag = (datetime.now().strftime('%Y_%m_%d_%H%M_%S'))
    filename = 'jexosim_intermediate_%s_%s'%(lab, time_tag)
    hdulist.writeto('%s/%s.fits'%(output_directory, filename))
    
    return '%s.fits'%(filename)