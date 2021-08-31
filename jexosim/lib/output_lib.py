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
from astropy.time import Time
from astropy import time, coordinates as coord

           
def write_record(opt, path, lab, input_text_file):
    
    textfile ='%s/%s.txt'%(path,lab)
    
    with open(textfile, "w") as f1:
        f1.write('===== Simulation values =====')
        f1.write('\n')
        f1.write('\nPlanet:  %s'%(opt.planet.planet.name))
        f1.write('\nChannel:  %s'%(opt.channel.name))
        if opt.simulation.sim_mode.val == 3:
            f1.write('\n\nNoise option:  noise budget')          
        else:
            f1.write('\n\nNoise option:  %s'%(opt.noise_tag))     
        f1.write('\n')

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
        f1.write('\n')
        if opt.simulation.sim_output_type.val == 1: # excludes fits files
            f1.write('\nApFactor:  %s'%(opt.pipeline.pipeline_ap_factor.val) )
            f1.write('\nAperture shape:  %s'%(opt.pipeline.pipeline_ap_shape.val) )
            f1.write('\nSpectral binning:  %s'%(opt.pipeline.pipeline_binning.val) )
            if opt.pipeline.pipeline_binning.val =='R-bin':
                        f1.write('\nBinned R power:  %s'%(opt.pipeline.pipeline_R.val) )
            else:
                      f1.write('\nBin size (pixels):  %s'%(opt.pipeline.pipeline_bin_size.val) )
        f1.write('\nWavelength: %s %s'%(opt.channel.pipeline_params.wavrange_lo.val, opt.channel.pipeline_params.wavrange_hi.val) )
    
    with open(textfile, "a") as f1:
        f1.write('\n')
        f1.write('\n')
        f1.write('===== Copy of input parameters file used =====')
        f1.write('\n')
        f1.write('\n')
      
    with open(input_text_file) as f:
        with open(textfile, "a") as f1:
            for line in f:       
                f1.write(line)
    f1.close()
    
    
def write_record_no_pipeline(opt, path, lab, input_text_file):
    
    textfile ='%s/%s.txt'%(path,lab)
    
    with open(textfile, "w") as f1:
        f1.write('===== Simulation values =====')
        f1.write('\n')
        f1.write('\nPlanet:  %s'%(opt.planet.planet.name))
        f1.write('\nChannel:  %s'%(opt.channel.name))
        if opt.simulation.sim_mode.val == 3:
            f1.write('\n\nNoise option:  noise budget')          
        else:
            f1.write('\n\nNoise option:  %s'%(opt.noise_tag))     
        f1.write('\n')

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
        f1.write('\n')
    
    with open(textfile, "a") as f1:
        f1.write('\n')
        f1.write('\n')
        f1.write('===== Copy of input parameters file used =====')
        f1.write('\n')
        f1.write('\n')
      
    with open(input_text_file) as f:
        with open(textfile, "a") as f1:
            for line in f:       
                f1.write(line)
    f1.close()


def write_to_fits(opt):   
    
    jexosim_msg('Save to fits file ...', 1)
    output_directory  = opt.common.output_directory.val


    hdu        = fits.PrimaryHDU()
    hdu.header['NEXP'] = (opt.n_exp,'Number of exposures')
    hdu.header['MACCUM_P'] = (opt.projected_multiaccum,'Multiaccum (projected)')
    hdu.header['MACCUM_E'] = (opt.effective_multiaccum,'Multiaccum (effective)')
    hdu.header['TEXP'] = (opt.exposure_time.value,'Duration of each intergration cycle [s]')
    hdu.header['PLANET'] = (opt.planet.planet.name,'Planet name')
    hdu.header['STAR'] = (opt.planet.planet.star.name,'Star name')
    
    hdu.header['CDELT1'] = (opt.channel.detector_pixel.plate_scale_x.val.to(u.deg).value,'Degrees/pixel')
    hdu.header['CDELT2'] = (opt.channel.detector_pixel.plate_scale_y.val.to(u.deg).value,'Degrees/pixel')
    hdulist = fits.HDUList(hdu)

    for i in range(opt.n_exp):
        for j in range(opt.effective_multiaccum):
            hdu = fits.ImageHDU(opt.data[..., i*opt.effective_multiaccum + j].value.astype(np.float32), name ='NDR')
            hdu.header['EXP'] = (i,'Exposure Number')
            hdu.header['NDR'] = (j,'NDR Number')
            hdu.header['EXTNAME'] = ('DATA')
            hdu.header['UNITS'] = ('%s'%(opt.data.unit) )
            hdulist.append(hdu)

    col1 = fits.Column(name='Wavelength {:s}'.format(opt.x_wav_osr[1::3].unit), format='E', 
 		       array=opt.x_wav_osr[1::3].value)
    col2 = fits.Column(name='Input Spectrum', format='E', 
 		       array=opt.planet.sed.sed[1::3].value)
    cols = fits.ColDefs([col1, col2])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.name ='INPUTS'
    hdulist.append(tbhdu)

    hdu = fits.ImageHDU(opt.pointing_timeline, name ='POINTING_TIMELINE')
    hdulist.append(hdu)
    
    hdu = fits.ImageHDU(opt.qe_grid, name ='PRNU')
    hdulist.append(hdu)
    
    col1 = fits.Column(name='Time {:s}'.format(opt.ndr_end_time.unit), format='E', 
 		       array=opt.ndr_end_time.value)

    cols = fits.ColDefs([col1])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.name ='TIME'
    hdulist.append(tbhdu)
 
    #write hdulist
    lab ='%s_%s'%(opt.observation.obs_channel.val, opt.exosystem_params.planet_name.val)
    time_tag = (datetime.now().strftime('%Y_%m_%d_%H%M_%S'))
    filename ='jexosim_%s_%s'%(lab, time_tag)
    hdulist.writeto('%s/%s.fits'%(output_directory, filename))
    
    return'%s.fits'%(filename)
    
    
def write_to_fits_intermediate(opt):   
    
    jexosim_msg('Saving binned light curves to fits file ...', 1)
    output_directory  = opt.common.output_directory.val

    hdu        = fits.PrimaryHDU()
    hdu.header['NEXP'] = (opt.n_exp,'Number of exposures')
    hdu.header['MACCUM_P'] = (opt.projected_multiaccum,'Multiaccum (projected)')
    hdu.header['MACCUM_E'] = (opt.effective_multiaccum,'Multiaccum (effective)')
    hdu.header['TEXP'] = (opt.exposure_time.value,'Duration of each intergration cycle [s]')
    hdu.header['PLANET'] = (opt.planet.planet.name,'Planet name')
    hdu.header['STAR'] = (opt.planet.planet.star.name,'Star name')

    hdulist = fits.HDUList(hdu)

    hdu = fits.ImageHDU(opt.pipeline_stage_1.binnedLC.value.astype(np.float32))
    hdu.header['EXTNAME'] = ('BINNED_LIGHT_CURVES')
    hdu.header['UNITS'] = ('%s'%(opt.pipeline_stage_1.binnedLC.unit) )
    hdulist.append(hdu)

    col1 = fits.Column(name='Wavelength {:s}'.format(opt.pipeline_stage_1.binnedWav.unit), format='E', 
 		       array=opt.pipeline_stage_1.binnedWav.value)
    cols = fits.ColDefs([col1])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.name ='WAVELENGTH'
    hdulist.append(tbhdu)

    col1 = fits.Column(name='Time {:s}'.format(opt.pipeline_stage_1.exp_end_time_grid.unit), format='E', 
 		       array=opt.pipeline_stage_1.exp_end_time_grid.value)

    cols = fits.ColDefs([col1])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.name ='TIME'
    hdulist.append(tbhdu)
 
    #write hdulist
    lab ='%s_%s'%(opt.observation.obs_channel.val, opt.exosystem_params.planet_name.val)
    time_tag = (datetime.now().strftime('%Y_%m_%d_%H%M_%S'))
    filename ='jexosim_intermediate_%s_%s'%(lab, time_tag)
    hdulist.writeto('%s/%s.fits'%(output_directory, filename))
    
    return'%s.fits'%(filename)





def write_to_fits_dev(opt):   
    
    jexosim_msg('Save to fits file ...', 1)
    output_directory  = opt.common.output_directory.val
    
    lab ='%s_%s'%(opt.observation.obs_channel.val, opt.exosystem_params.planet_name.val)
    time_tag = (datetime.now().strftime('%Y_%m_%d_%H%M_%S'))
    filename ='jexosim_%s_%s'%(lab, time_tag)
    
    ut = Time(datetime.utcnow(), scale='utc')
    t = Time(ut, format='fits')
    
    exp_end_time = t + opt.exposure_time*opt.n_exp
    exp_start_time = t  
    exp_mid_time = t +   opt.exposure_time*opt.n_exp/2
    
# =============================================================================

# =============================================================================
    hdu        = fits.PrimaryHDU()
    hdu.header['DATE'] =  (t.value,'UTC date file created')
    hdu.header['ORIGIN'] = ('JexoSim','Organization responsible for creating file')
    hdu.header['FILENAME'] = (filename,'Name of the file')
    hdu.header['FILETYPE'] = ('raw','Type of data in the file')
    hdu.header['DATAMODL'] = ('RampModel','Type of data model')
    hdu.header['TELESCOP'] = ('JWST','Telescope used to acquire the data')

    hdu.header['TITLE'] = ('xxxx','Proposal title')
    hdu.header['PI_NAME'] = ('xxxx','Principal investigator name')
    hdu.header['CATEGORY'] = ('xxxx','Program category')
    hdu.header['SUBCAT'] = ('UNKNOWN','Program sub-category')
    hdu.header['SCICAT'] = ('Exoplanets and Exoplanet Formation','Science category assigned durin')
    hdu.header['CONT_ID'] = (0,'Continuation of previous program') 

    hdu.header['DATE-OBS'] = (str(ut.value.date()),'[yyyy-mm-dd] UTC date at start of exposure')    
    hdu.header['TIME-OBS'] = (str(ut.value.time()),'[hh:mm:ss.sss] UTC time at start of exposure')  
    hdu.header['OBS_ID'] = ('xxxx','Programmatic observation identifier')    
    hdu.header['VISIT_ID'] = ('xxxx','Visit identifier')                               
    hdu.header['PROGRAM'] = ('xxxx','Program number')                                 
    hdu.header['OBSERVTN'] = ('001','Observation number')                             
    hdu.header['VISIT'] = ('001','Visit number')                                   
    hdu.header['VISITGRP'] = ('01','Visit group identifier')                         
    hdu.header['SEQ_ID'] = ('1','Parallel sequence identifier')                  
    hdu.header['ACT_ID'] = ('01','Activity identifier')                            
    hdu.header['EXPOSURE'] = ('00001','Exposure request number')                        
    hdu.header['OBSLABEL'] = ('Baseline GRISMR','Proposer label for the observation')   

    hdu.header['TSOVISIT'] = (True,'Time Series Observation visit indicator')    

    hdu.header['TARGPROP'] = (opt.model_exosystem.planet_name.val, " Proposer's name for the target" )               
    hdu.header['TARGNAME'] = ('UNKNOWN','Standard astronomical catalog name for target') 
    hdu.header['TARG_RA'] = (opt.model_exosystem.ra.val,'Target RA at mid time of exposure')              
    hdu.header['TARG_DEC'] = (opt.model_exosystem.dec.val,'Target Dec at mid time of exposure')   

    hdu.header['INSTRUME']= ('NIRCAM','Instrument used to acquire the data')            
    hdu.header['DETECTOR']= ('NRCALONG','Name of detector used to acquire the data')      
    hdu.header['MODULE']= ('A','NIRCam module: A or B')                          
    hdu.header['CHANNEL']= ('LONG','Instrument channel')                             
    hdu.header['FILTER']= ('F322W2','Name of the filter element used')                
    hdu.header['PUPIL']= ('GRISMR','Name of the pupil element used')                                                                                                          

    hdu.header['EXP_TYPE']= ('NRC_TSGRISM','Type of data in the exposure')    
    hdu.header['EXPSTART']=(exp_start_time.mjd,'UTC exposure start time')                         
    hdu.header['EXPMID']=(exp_mid_time.mjd,'UTC exposure mid time')                           
    hdu.header['EXPEND']=(exp_end_time.mjd,'UTC exposure end time')                                          
    hdu.header['READPATT']=(opt.pattern,'Readout pattern')                                 
    hdu.header['NINTS']=(opt.n_exp,'Number of integrations in exposure')              
    hdu.header['INTSTART']=(1,'Starting integration number in this segment')     
    hdu.header['INTEND']=(opt.n_exp,'Ending integration number in this segment')       
    hdu.header['NGROUPS']=(opt.effective_multiaccum,'Number of groups in integration')                 
    hdu.header['NFRAMES']=(opt.nframes,'Number of frames per group')                     
    hdu.header['GROUPGAP']=(opt.nskip,'Number of frames dropped between groups')         
    hdu.header['TSAMPLE']=(10,'[us] Time between samples')                       
    hdu.header['TFRAME']=(opt.t_f.value,'[s] Time between frames')                         
    hdu.header['TGROUP']=(opt.t_g.value,'[s] Time between groups')                         
    hdu.header['EFFINTTM']=(opt.effective_multiaccum*opt.t_g.value,'[s] Effective integration time')                  
    hdu.header['EFFEXPTM']=(opt.n_exp*opt.effective_multiaccum*opt.t_g.value,'[s] Effective exposure time')                     
    hdu.header['DURATION']=(opt.exposure_time.value,'[s] Total duration of exposure')                  
    hdu.header['NRSTSTRT']=(0,'Number of resets at start of exposure')           
    hdu.header['NRESETS']=(opt.channel.detector_readout.nRST.val,'Number of resets between integrations')                                                                                                   

    hdu.header['SUBARRAY']=(opt.subarray_name,'Subarray used')                                   
    hdu.header['SUBSTRT1']=(1,'Starting pixel in axis 1 direction')              
    hdu.header['SUBSTRT2']=(1,'Starting pixel in axis 2 direction')              
    hdu.header['SUBSIZE1']=(opt.fp_signal[1::3,1::3].shape[1],'Number of pixels in axis 1 direction')            
    hdu.header['SUBSIZE2']=(opt.fp_signal[1::3,1::3].shape[0],'Number of pixels in axis 2 direction')            
    hdu.header['FASTAXIS']=(-1,'Fast readout axis direction')                     
    hdu.header['SLOWAXIS']=(2,'Slow readout axis direction')      
        
    hdu.header['PATTTYPE']= ('NONE','Primary dither pattern type')                    
    hdu.header['PATT_NUM']= (1,'Position number in primary pattern')             
    hdu.header['NUMDTHPT']= (1,'Total number of points in pattern')              
    hdu.header['PATTSIZE']= ('DEFAULT','Primary dither pattern size')                    
    hdu.header['SUBPXNUM']= (1,'Subpixel pattern number')                        
    hdu.header['SUBPXPNS']= (1,'Total number of points in subpixel pattern')     
    hdu.header['XOFFSET']= (0.0,'x offset from pattern starting position')        
    hdu.header['YOFFSET']= (1.449,'y offset from pattern starting position')                                                                                               
      
    hdu.header['APERNAME']= ('NRCA5_GRISM64_F322W2','PRD science aperture used')                     
    hdu.header['MRGEVRSN']= ('1.3.4.dev135+gfb3f8f0','Mirage version used')                           
    hdu.header['YAMLFILE']= ('/home/anadkarni/NIRCam_AZ_TSO_Data_Challenge/GJ436/GJ436_1110/GJ436_yaml_data/jw00042001001_01101_00002_nrca5.yaml','Mirage input yaml file')                                             
    hdu.header['GAIN']= (2.1923525,'Gain value used by Mirage')                        
    hdu.header['DISTORTN']= ('/home/anadkarni/crds_cache/references/jwst/nircam/jwst_nircam_distortion_0090.asdf','Distortion reffile used by Mirage')                                  
    hdu.header['IPC']= ('/home/anadkarni/crds_cache/references/jwst/nircam/Kernel_to_add_IPC_effects_from_jwst_nircam_ipc_0028.fits','IPC kernel used by Mirage')                                          
    hdu.header['PIXARMAP']= ('/home/anadkarni/crds_cache/references/jwst/nircam/jwst_nircam_area_0015.fits','Pixel area map used by Mirage')                                      
    hdu.header['CROSSTLK']= ('/home/anadkarni/anaconda3/envs/astro/lib/python3.6/site-packages/mirage/config/xtalk20150303g0.errorcut.txt','Crosstalk file used by Mirage')                                      
    hdu.header['FLUX_CAL']= ('/home/anadkarni/anaconda3/envs/astro/lib/python3.6/site-packages/mirage/config/NIRCam_zeropoints.list','Flux calibration file used by Mirage')                              
    hdu.header['FTHRUPUT']= ('/home/anadkarni/anaconda3/envs/astro/lib/python3.6/site-packages/mirage/config/placeholder.txt','Filter throughput file used by Mirage')                              
    hdu.header['PTSRCCAT']= ('None','Point source catalog used by Mirage')              
    hdu.header['GALAXCAT']= ('None','Galaxy source catalog used by Mirage')             
    hdu.header['EXTNDCAT']= ('None','Extended source catalog used by Mirage')           
    hdu.header['MTPTSCAT']= ('None','Moving point source catalog used by Mirage')      
    hdu.header['MTSERSIC']= ('None','Moving Sersic catalog used by Mirage')             
    hdu.header['MTEXTEND']= ('None','Moving extended target catalog used by Mirage')    
    hdu.header['NONSDRAL']= ('None','Non-Sidereal catalog used by Mirage')              
    hdu.header['BKGDRATE']= ('medium','Background rate used by Mirage')                   
    hdu.header['TRACKING']= ('sidereal','Telescope tracking type for Mirage')               
    hdu.header['POISSON']= (1472856351,'Random num generator seed for Poisson noise in')   
    hdu.header['PSFWFE']= ('predicted','WebbPSF Wavefront error used by Mirage')           
    hdu.header['PSFWFGRP']= (0,'WebbPSF wavefront error group used by Mirage')     
    hdu.header['CRLIB']= ('SUNMAX','Cosmic ray library used by Mirage')                
    hdu.header['CRSCALE']= (1.0,'Cosmic ray rate scaling factor used by Mirage')    
    hdu.header['CRSEED']= (2820613178,'Random number generator seed for cosmic rays in')  
    
    hdu.header.add_blank('', before ='TITLE')
    hdu.header.add_blank('Program information', before ='TITLE')
    hdu.header.add_blank('', before ='TITLE')
    hdu.header.add_blank('', before ='DATE-OBS')
    hdu.header.add_blank('Observational identifiers', before ='DATE-OBS')
    hdu.header.add_blank('', before ='DATE-OBS') 
    hdu.header.add_blank('', before ='TSOVISIT')
    hdu.header.add_blank('Visit information', before ='TSOVISIT') 
    hdu.header.add_blank('', before ='TSOVISIT')    
    hdu.header.add_blank('', before ='INSTRUME')
    hdu.header.add_blank('Instrument configuration information', before ='INSTRUME')      
    hdu.header.add_blank('', before ='INSTRUME') 
    hdu.header.add_blank('', before ='EXP_TYPE')
    hdu.header.add_blank('Exposure parameters', before ='EXP_TYPE')    
    hdu.header.add_blank('', before ='EXP_TYPE')
    hdu.header.add_blank('', before ='SUBARRAY')
    hdu.header.add_blank('Subarray parameters', before ='SUBARRAY')      
    hdu.header.add_blank('', before ='SUBARRAY')  
    hdu.header.add_blank('', before ='PATTTYPE')
    hdu.header.add_blank('Dither information', before ='PATTTYPE')    
    hdu.header.add_blank('', before ='PATTTYPE')
    hdu.header.add_blank('', before='APERNAME')
    hdu.header.add_blank('Aperture information', before='APERNAME')      
    hdu.header.add_blank('', before='APERNAME')  
    hdu.header.add_blank('', before ='TARGPROP')
    hdu.header.add_blank('Target information', before ='TARGPROP')
    hdu.header.add_blank('', before ='TARGPROP') 
    
    hdulist = fits.HDUList(hdu)
# =============================================================================
#   SCI array
# =============================================================================
    img_array = np.zeros((opt.n_exp, opt.effective_multiaccum, opt.data.shape[0], opt.data.shape[1]))
    
    ct = 0
    for i in range(opt.n_exp):
        for j in range(opt.effective_multiaccum):
            img_array[i][j] = opt.data[...,ct]
            ct+=1
            
    hdu = fits.ImageHDU(img_array.astype(np.float32), name ='SCI')
    hdu.header['EXTNAME'] = ('SCI')
    hdu.header.set('BSCALE', 1, before='EXTNAME')
    hdu.header.set('BZERO', 0, before='EXTNAME')
    
    hdu.header['RADESYS'] = ('ICRS','Name of the coordinate reference frame')                                                                                                 
                                                                                                                      
    hdu.header['RA_V1'] = (opt.model_exosystem.ra.val,'[deg] RA of telescope V1 axis')                  
    hdu.header['DEC_V1'] = (opt.model_exosystem.ra.val,'[deg] Dec of telescope V1 axis')                 
    hdu.header['PA_V3'] = (0.0,'[deg] Position angle of telescope V3 axis')                                                                                              
                                                                                                                                       
    hdu.header['WCSAXES'] = (2,'number of World Coordinate System axes')         
    hdu.header['CRPIX1'] = (opt.fp_signal[1::3,1::3].shape[1]/2,'axis 1 coordinate of the reference pixel')       
    hdu.header['CRPIX2'] = (opt.fp_signal[1::3,1::3].shape[0]/2,'axis 2 coordinate of the reference pixel')       
    hdu.header['CRVAL1'] = (opt.model_exosystem.ra.val,'first axis value at the reference pixel')        
    hdu.header['CRVAL2'] = (opt.model_exosystem.dec.val,'second axis value at the reference pixel')       
    hdu.header['CTYPE1'] = ('RA---TAN','first axis coordinate type')                     
    hdu.header['CTYPE2'] = ('DEC--TAN','second axis coordinate type')                    
    hdu.header['CUNIT1'] = ('deg','first axis units')                               
    hdu.header['CUNIT2'] = ('deg','second axis units')                              
    hdu.header['CDELT1'] = (opt.channel.detector_pixel.plate_scale_x.val.value, 'first axis increment per pixel')                 
    hdu.header['CDELT2'] = (opt.channel.detector_pixel.plate_scale_y.val.value, 'second axis increment per pixel')                
    hdu.header['V2_REF'] = (50.767712,'[arcsec] Telescope v2 coordinate of the referen')
    hdu.header['V3_REF'] = (-556.476341,'[arcsec] Telescope v3 coordinate of the referen')
    hdu.header['VPARITY'] = (-1,'Relative sense of rotation between Ideal xy and')
    hdu.header['V3I_YANG'] = (0.22580389,'[deg] Angle from V3 axis to Ideal y axis')       
    hdu.header['ROLL_REF'] = (0.007083997614010916,'[deg] V3 roll angle at the ref point (N over E)')
    hdu.header['XREF_SCI'] = (opt.fp_signal[1::3,1::3].shape[1]/2,'Aperture X reference point in SCI frame')        
    hdu.header['YREF_SCI'] = (opt.fp_signal[1::3,1::3].shape[0]/2,'Aperture Y reference point in SCI frame')        
    hdu.header['EXTVER'] = (1,'extension value')                                
    hdu.header['PC1_1'] = (-0.9999917392791804)                                                 
    hdu.header['PC1_2'] = (0.004064649234531177)                                                  
    hdu.header['PC2_1'] = (0.004064649234531177)                                                  
    hdu.header['PC2_2'] = (0.9999917392791804)                                                  
    hdu.header['RA_REF'] = (opt.model_exosystem.ra.val)                                                 
    hdu.header['DEC_REF'] = (opt.model_exosystem.dec.val)                                                  
                                                                                           
    hdu.header.add_blank('', before ='RADESYS')
    hdu.header.add_blank('Information about the coordinates in the file', before ='RADESYS')
    hdu.header.add_blank('', before ='RADESYS') 
    
    hdu.header.add_blank('', before ='RA_V1')
    hdu.header.add_blank('Spacecraft pointing information', before ='RA_V1')   
    hdu.header.add_blank('', before ='RA_V1')
    
    hdu.header.add_blank('', before ='WCSAXES')
    hdu.header.add_blank('WCS parameters', before ='WCSAXES')
    hdu.header.add_blank('', before ='WCSAXES') 


   
    hdulist.append(hdu)
    
# =============================================================================
#   ZEROFRAME array
# =============================================================================
    
    img_array = np.zeros((opt.n_exp, opt.data.shape[0], opt.data.shape[1]))
    
    ct = 0
    for i in range(opt.n_exp):
            img_array[i]= opt.data[...,ct]
            ct+=opt.effective_multiaccum
            
    hdu = fits.ImageHDU(img_array.astype(np.float32), name ='ZEROFRAME')
    hdu.header['EXTNAME'] = ('ZEROFRAME')
    hdu.header.set('BSCALE', 1, before='EXTNAME')
    hdu.header.set('BZERO', 0, before='EXTNAME')
    hdu.header['EXTVER'] = (1 , 'extension value')
 
    hdulist.append(hdu)


# =============================================================================
#   GROUP
# =============================================================================
    int_number = np.repeat(np.arange(1,opt.n_exp+1, 1), opt.effective_multiaccum)
    gp_number = np.arange(1,opt.n_exp*opt.effective_multiaccum+1, 1)
    end_day = np.ones(len(gp_number))*10
    end_milliseconds = opt.ndr_end_time.value*1000
    end_submilliseconds = np.ones(len(gp_number))*0
    
    ip_peg = coord.SkyCoord(opt.model_exosystem.ra.val, opt.model_exosystem.dec.val,
      unit=(u.deg, u.deg), frame='icrs')
    
    greenwich = coord.EarthLocation.of_site('greenwich')  
    
    int_end_time = t+opt.ndr_end_time[opt.effective_multiaccum-1::opt.effective_multiaccum]
    times = time.Time(int_end_time, format='mjd',
                    scale='utc', location=greenwich)    
    ltt_bary = times.light_travel_time(ip_peg)  
    time_barycentre = times.tdb + ltt_bary.value
    int_end_time_mjd = np.double(times.value)
    int_end_time_bjd = np.double(time_barycentre.value)
      
     
    aa  = np.add.reduceat(opt.duration_per_ndr, np.arange(0, len(opt.duration_per_ndr), opt.effective_multiaccum)) 
    int_start_time = int_end_time - aa
    times = time.Time(int_start_time, format='mjd',
                    scale='utc', location=greenwich)     
    ltt_bary = times.light_travel_time(ip_peg) 
    time_barycentre = times.tdb + ltt_bary.value
    int_start_time_mjd = np.double(times.value)
    int_start_time_bjd = np.double(time_barycentre.value)
    
   
    int_mid_time = int_end_time - aa/2
    times = time.Time(int_mid_time, format='mjd',
                    scale='utc', location=greenwich)     
    ltt_bary = times.light_travel_time(ip_peg) 
    time_barycentre = times.tdb + ltt_bary.value
    int_mid_time_mjd = np.double(times.value)
    int_mid_time_bjd = np.double(time_barycentre.value)
  
 
 
    group_end_time = t+opt.ndr_end_time
    group_end_time0 = group_end_time.value
    times = time.Time(group_end_time, format='mjd',
                    scale='utc', location=greenwich)  
    ltt_bary = times.light_travel_time(ip_peg)  
    ltt_helio = times.light_travel_time(ip_peg,'heliocentric') 
    time_bary = times.tdb + ltt_bary  
    time_helio = times.utc + ltt_helio
    time_bary = np.double(time_bary.value)
    time_helio = np.double(time_helio.value)
 
    number_of_columns = np.ones(len(gp_number))*opt.data.shape[1]
    number_of_rows = np.ones(len(gp_number))*opt.data.shape[0]
    number_of_gaps = np.ones(len(gp_number))*0
    completion_code_number = np.ones(len(gp_number))*0
    completion_code_text = np.repeat('Normal Completion', len(gp_number))


    col1 = fits.Column(name='integration_number', format='I', array = int_number)
    col2 = fits.Column(name='group_number', format='I', array = gp_number)
    
    col3 = fits.Column(name='end_day', format='I', array = end_day)
    col4 = fits.Column(name='end_millisecond', format='J', array = end_milliseconds)
    col5 = fits.Column(name='sub_millisecond', format='I', array = end_submilliseconds)
    col6 = fits.Column(name='group_end_time', format='26A', array = group_end_time0)

    
    col7 = fits.Column(name='number_of_columns', format='I', array = number_of_columns)
    col8 = fits.Column(name='number_of_rows', format='I', array = number_of_rows)
    col9 = fits.Column(name='number_of_gaps', format='I', array = number_of_gaps)

    
    col10 = fits.Column(name='completion_code_number', format='I', array = completion_code_number)
    col11 = fits.Column(name='completion_code_text', format='36A', array = completion_code_text)
    
    col12 = fits.Column(name='bary_end_time', format='D', array = time_bary)
    col13 = fits.Column(name='helio_end_time', format='D', array = time_helio)
 
    
    cols = fits.ColDefs([col1, col2,col3, col4, col5, col6, col7, 
                         col8, col9, col10, col11, col12, col13])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.name ='GROUP'
    tbhdu.header.set('EXTVER', 1, 'extension value', before='TTYPE1')
    tbhdu.header.set('EXTNAME', 'GROUP', before='EXTVER')
    hdulist.append(tbhdu)
    

    
# =============================================================================
#     INT_TIMES
# =============================================================================
    int_number =  np.arange(1,opt.n_exp+1, 1) 
    col1 = fits.Column(name='integration_number', format='J', array = int_number)
    col2 = fits.Column(name='int_start_MJD_UTC', format='D', array = int_start_time_mjd)
    col3 = fits.Column(name='int_mid_MJD_UTC', format='D', array = int_mid_time_mjd)
    col4 = fits.Column(name='int_end_MJD_UTC', format='D', array = int_end_time_mjd)
    col5 = fits.Column(name='int_start_BJD_TDB', format='D', array = int_start_time_bjd)
    col6 = fits.Column(name='int_mid_BJD_TDB', format='D', array = int_mid_time_bjd)
    col7 = fits.Column(name='int_end_BJD_TDB', format='D', array = int_end_time_bjd)

    cols = fits.ColDefs([col1, col2,col3, col4, col5, col6, col7])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.name ='INT_TIMES'
    tbhdu.header.set('EXTVER', 1, 'extension value', before='TTYPE1')
    tbhdu.header.set('EXTNAME', 'INT_TIMES', before='EXTVER')
    hdulist.append(tbhdu)
    
# =============================================================================
#     ASDF
# =============================================================================
    asdf = np.ones(1) # not sure if this is needed
    col1 = fits.Column(name='ASDF_METADATA', format='6718B', array = asdf)
    cols = fits.ColDefs([col1])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.name ='ASDF'
    hdulist.append(tbhdu)

    #write hdulist

    hdulist.writeto('%s/%s.fits'%(output_directory, filename))
    
    return'%s.fits'%(filename)
    