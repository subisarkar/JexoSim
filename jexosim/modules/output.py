"""
JexoSim 
2.0
Output module
v1.0

"""

from astropy.io import fits
from astropy import units as u
import os, glob
from jexosim.lib import jexosim_lib 
from jexosim.lib.jexosim_lib import jexosim_msg
from datetime import datetime
import numpy as np

def run(opt):
    jexosim_msg('Save to fits file ... ', 1)
    output_directory  = opt.common.output_directory.val

  # existing_sims = glob.glob(os.path.join(out_path, 'sim_*'))
  
  # if not existing_sims:
  #   # Empty list
  #   sim_number = 0
  # else:
  #   sim_number = sorted([np.int(l.split('_')[-1]) for l in existing_sims])[-1]
  #   sim_number += 1
  
  # out_path = os.path.join(out_path, 'sim_{:04d}'.format(sim_number))

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
    col2 = fits.Column(name='Input Contrast Ratio', format='E', 
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
    filename = 'JexoSim_%s_%s'%(lab, time_tag)
    hdulist.writeto('%s/%s.fits'%(output_directory, filename))
   

  
    # textfile = (os.path.join(out_path, '{:s}_info.txt'.format(key)))
    # file = open(textfile,'w')
    # file.write('Planet:  %s'%(opt.astroscene.planet.val))    
#    file.write('\nChannel:  %s'%(channel.keys()[0]))   
#    file.write('\n\nNoise option:  %s'%(opt.noise_option)) 
#    file.write('\n\nSource?:  %s'%(bool(opt.source_switch)))    
#    file.write('\nSN:  %s'%(opt.noise.EnableShotNoise.val))
#    file.write('\nRN:  %s'%(opt.noise.EnableReadoutNoise.val))    
#    file.write('\nJN(Spatial):  %s'%(opt.noise.EnableSpatialJitter.val))
#    file.write('\nJN(Spectral):  %s'%(opt.noise.EnableSpectralJitter.val))
#    file.write('\nEmmission?:  %s'%(bool(opt.emm_switch)))
#    file.write('\nZodi?:  %s'%(bool(opt.zodi_switch)))
#    file.write('\nDC?:  %s'%(bool(opt.dc_switch)))
#    file.write('\nRandom QE grid selected?:  %s'%(bool(opt.use_random_QE)))    
#    file.write('\nQE grid applied?:  %s'%(bool(opt.apply_QE)))
#    file.write('\nFlat field and bkg subtraction applied?:  %s'%(bool(opt.apply_flat)))
#    file.write('\nDiffuse source?:  %s'%(bool(opt.diff)))    
#    file.write('\n ')   
#    file.write('\nmultiaccum:  %s'%(opt.detector_readout.multiaccum.val))  
#    file.write('\nNDR_time (>NDR0):  %s'%(opt.NDR_time)) 
#    file.write('\nNDR_time (NDR0):  %s'%(opt.detector_readout.nNDR0()*opt.frame_time) )    
#    file.write('\nexposure cycle time:  %s'%(opt.exposure_time) ) 
#    file.write('\nnumber of exposures:  %s'%(opt.n_exp) ) 
#    file.write('\ncalculated integration time (this is used to find the exposure time):  %s'%(opt.integration_time_estimate))

    # file.close() 
    

