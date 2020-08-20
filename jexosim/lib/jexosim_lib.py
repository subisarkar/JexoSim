"""
JexoSim 
v1.0
JexoSim library
v1.0

"""

import numpy as np
from scipy import interpolate
from scipy.integrate import cumtrapz
import scipy.special
import sys, os
import scipy.constants as spc
from astropy import units as u
import matplotlib.pyplot as plt
plt.style.use('classic')


def calc_logg(m,r):
    m = m.to(u.kg)
    r = r.to(u.m)
    G = spc.G*u.m**3/u.kg**1/u.s**2
    g = G*m /r**2
    g0 = g.to(u.cm/u.s**2)
    return g, np.log10(g0.value)
    
def calc_EqT(T_s,R_s,a,A,tidal):   
    if tidal == 1: 
        factor = 2.
    else:
        factor = 4.
    T_p = T_s*(R_s.to(u.m)/a.to(u.m))**0.5*((1-A)/factor)**0.25
     
    return T_p       

def jexosim_error(error_msg):
    sys.stderr.write("Error code: {:s}\n".format(error_msg))
    sys.exit(0)
    
def jexosim_msg(msg, show):
  if show == 1:
      print (msg)
      
      
def jexosim_plot(figure_name, show, image=False, xlabel=None, 
                 ylabel=None, image_data=None, xdata=None, ydata=None, marker='-',
                 xlim=None, ylim=None, aspect=None, interpolation=None, linewidth=1, alpha=1,
                 grid=False, label=None):
  if hasattr(image_data, "unit"): # fid for astropy image issue with units
      image_data = image_data.value
  if show == 1:
      if image == True:
          plt.figure(figure_name)
          if aspect or interpolation:
              plt.imshow(image_data, aspect=aspect, interpolation=None)
          else:    
              plt.imshow(image_data)
          if xlabel:
              plt.xlabel(xlabel)
          if ylabel:
              plt.ylabel(ylabel)         
              
      else:
          plt.figure(figure_name)
 
          if xdata is not None:
                plt.plot(xdata, ydata, marker, linewidth=linewidth, alpha=alpha, label=label)
          else:
                plt.plot(ydata, marker,linewidth=linewidth, alpha=alpha, label=label)
                        
          if xlabel:
              plt.xlabel(xlabel)
          if ylabel:
              plt.ylabel(ylabel)
          if ylim:
              plt.ylim(ylim[0],ylim[1])
          if xlim:
              plt.xlim(xlim[0],xlim[1])
          if grid==True:
              plt.grid()
          if label is not None:
              plt.legend(loc='best')

    
def calc_EqT(T_s,R_s,a,A,tidal):
   
    if tidal == 1: 
        factor = 2.
    else:
        factor = 4.
    T_p = T_s*(R_s.to(u.m)/a.to(u.m))**0.5*((1-A)/factor)**0.25
     
    return T_p       

def jexosim_error(error_msg):
    sys.stderr.write("Error code: {:s}\n".format(error_msg))
    sys.exit(0)
    
  
def logbin(x, a,  R, xmin=None, xmax=None):
  n = a.size
  imin = 0
  imax = n-1
  
  if xmin == None or xmin < x.min(): xmin = x.min()
  if xmax == None or xmax > x.max(): xmax = x.max()
  
  idx = np.argsort(x)
  xp = x[idx]
  yp = a[idx]
  
  delta_x = xmax/R
  N = 20.0 * (xmax-xmin)/delta_x
  _x = np.linspace(xmin,xmax, N)
  _y = np.interp(_x, xp, yp)
  
  nbins = 1+np.round( np.log(xmax/xmin)/np.log(1.0 + 1.0/R) ).astype(np.int)
  bins = xmin*np.power( (1.0+1.0/R), np.arange(nbins))
  
  slices  = np.searchsorted(_x, bins)
  counts = np.ediff1d(slices)
  
  mean = np.add.reduceat(_y, slices[:-1])/(counts)
  bins = 0.5*(bins[:-1] + bins[1:])
  return bins[:-1], mean[:-1]

def rebin(x, xp, fp):
  ''' Resample a function fp(xp) over the new grid x, rebinning if necessary, 
    otherwise interpolates
    Parameters
    ----------
    x	: 	array like
	New coordinates
    fp 	:	array like
	y-coordinates to be resampled
    xp 	:	array like
	x-coordinates at which fp are sampled
	
    Returns
    -------
    out	: 	array like
	new samples
  
  '''
  
  if (x.unit != xp.unit):
    print (x.unit, xp.unit)
    jexosim_error('Units mismatch')
  
  idx = np.where(np.logical_and(xp > 0.9*x.min(), xp < 1.1*x.max()))[0]
  xp = xp[idx]
  fp = fp[idx]
  
  if np.diff(xp).min() < np.diff(x).min():
    # Binning!
    c = cumtrapz(fp, x=xp)
    xpc = xp[1:]
        
    delta = np.gradient(x)
    new_c_1 = np.interp(x-0.5*delta, xpc, c, 
                        left=0.0, right=0.0)
    new_c_2 = np.interp(x+0.5*delta, xpc, c, 
                        left=0.0, right=0.0)
    new_f = (new_c_2 - new_c_1)/delta
 
  else:
    # Interpolate !
    new_f = np.interp(x, xp, fp, left=0.0, right=0.0)
    
  new_f = (new_f.value)*fp.unit
    
#    func = interpolate.interp1d(xp, fp, kind='quadratic', bounds_error=None, fill_value=0.0)
#    new_f  = func(x)*fp.unit
  '''
  import matplotlib.pyplot as plt
  plt.plot(xp, fp, '-')
  plt.plot(x, new_f, '.-')
  plt.show()
  # check
  print np.trapz(new_f, x)
  idx = np.where(np.logical_and(xp>= x.min(), xp <= x.max()))
  print np.trapz(fp[idx], xp[idx])
  '''
  return x, new_f
  
def fast_convolution(im, delta_im, ker, delta_ker):
  """ fast_convolution.
    Convolve an image with a kernel. Image and kernel can be sampled on different
      grids defined.
    
    Parameters
    __________
      im : 			array like
				the image to be convolved
      delta_im :		scalar
				image sampling interval
      ker : 			array like
				the convolution kernel
      delta_ker :		scalar
				Kernel sampling interval
    Returns
    -------
      spectrum:			array like
				the image convolved with the kernel.
  """
  fc_debug = False
  # Fourier transform the kernel
  kerf = (np.fft.rfft2(ker))
  ker_k = [ np.fft.fftfreq(ker.shape[0], d=delta_ker),
	   np.fft.rfftfreq(ker.shape[1], d=delta_ker) ]
  ker_k[0] = np.fft.fftshift(ker_k[0])
  kerf     = np.fft.fftshift(kerf, axes=0)
  
  # Fourier transform the image
  imf  = np.fft.rfft2(im)
  im_k = [ np.fft.fftfreq(im.shape[0], d=delta_im),
	   np.fft.rfftfreq(im.shape[1], d=delta_im) ]
  im_k[0] = np.fft.fftshift(im_k[0])
  imf     = np.fft.fftshift(imf, axes=0)
  
  # Interpolate kernel 
  kerf_r = interpolate.RectBivariateSpline(ker_k[0], ker_k[1],
					   kerf.real)
  kerf_i = interpolate.RectBivariateSpline(ker_k[0], ker_k[1],
					   kerf.imag)
  if (fc_debug):
    pl.plot(ker_k[0], kerf[:, 0].real,'.r')
    pl.plot(ker_k[0], kerf[:, 0].imag,'.g')
    pl.plot(im_k[0], kerf_r(im_k[0], im_k[1])[:, 0],'-r')
    pl.plot(im_k[0], np.abs(imf[:, 0]),'-b')

  # Convolve
  imf = imf * (kerf_r(im_k[0], im_k[1]) + 1j*kerf_i(im_k[0], im_k[1])) 
  
  if (fc_debug):
    pl.plot(im_k[0], np.abs(imf[:, 0]),'-y')

  imf = np.fft.ifftshift(imf, axes=0)
  
  return np.fft.irfft2(imf)*(delta_ker/delta_im)**2

   
def planck(wl, T):
  """ Planck function. 
    
    Parameters
    __________
      wl : 			array
				wavelength [micron]
      T : 			scalar
				Temperature [K]
				Spot temperature [K]
    Returns
    -------
      spectrum:			array
				The Planck spectrum  [W m^-2 sr^-2 micron^-1]
  """
    
  a = np.float64(1.191042768e8)*u.um**5 *u.W/ u.m**2 /u.sr/u.um
  b = np.float64(14387.7516)*1*u.um * 1*u.K
  try:
    x = b/(wl*T)
    bb = a/wl**5 / (np.exp(x) - 1.0)
  except ArithmeticError:
    bb = np.zeros(np.size(wl))
  return bb
 
 
def sed_propagation(sed, transmission, emissivity=None, temperature = None):
  sed.sed = sed.sed*transmission.sed
  if emissivity and temperature:
    sed.sed = sed.sed + emissivity.sed*planck(sed.wl, temperature)

  return sed
  
  
def Psf(wl, fnum_x, fnum_y, delta, nzero = 4, shape='airy'):
  '''
  Calculates an Airy Point Spread Function arranged as a data-cube. The spatial axies are 
  0 and 1. The wavelength axis is 2. Each PSF area is normalised to unity.
  
  Parameters
  ----------
  wl	: ndarray [physical dimension of length]
    array of wavelengths at which to calculate the PSF
  fnum : scalar
    Instrument f/number
  delta : scalar
    the increment to use [physical units of length]
  nzero : scalar
    number of Airy zeros. The PSF kernel will be this big. Calculated at wl.max()
  shape : string
    Set to 'airy' for a Airy function,to 'gauss' for a Gaussian
  
  Returns
  ------
  Psf : ndarray
    three dimensional array. Each PSF normalised to unity
  '''
#  fnum_y =  fnum_x
  delta = delta.to(wl.unit)
  Nx = int(np.round(scipy.special.jn_zeros(1, nzero)[-1]/(2.0*np.pi) * fnum_x*wl.max()/delta).astype(np.int))
  
  Ny = Nx = np.int(Nx)
#  Ny = int(np.round(scipy.special.jn_zeros(1, nzero)[-1]/(2.0*np.pi) * fnum_y*wl.max()/delta).astype(np.int))

  if shape=='airy':
    d = 1.0/(fnum_x*(1.0e-30*delta.unit+wl))
  elif shape=='gauss':
    sigma = 1.029*fnum_x*(1.0e-30*delta.unit+wl)/np.sqrt(8.0*np.log(2.0))
    d     = 0.5/sigma**2
    
  x = np.linspace(-Nx*delta.item(), Nx*delta.item(), 2*Nx+1)*delta.unit
  y = np.linspace(-Ny*delta.item(), Ny*delta.item(), 2*Ny+1)*delta.unit
  
  yy, xx = np.meshgrid(y, x)
 
  if shape=='airy':
    arg = 1.0e-20+np.pi*np.multiply.outer(np.sqrt(yy**2 + xx**2), d)
    img   = (scipy.special.j1(arg)/arg)**2
  elif shape=='gauss':
    arg = np.multiply.outer(yy**2 + xx**2, d)
    img = np.exp(-arg)
  
  if fnum_y !=  fnum_x:
      x_pix = img.shape[0]  
      stretch = fnum_y/fnum_x
      y_pix = int(np.round(x_pix*stretch))
      
      x_pos = np.linspace(0,1,x_pix)
      y_pos = np.linspace(0,1,y_pix)
      new_img = np.zeros((y_pix, x_pix, img.shape[2]))
   
      for i in range(img.shape[2]):
          img_ = interpolate.interp2d(x_pos,x_pos,img[...,i], kind='linear')(x_pos,y_pos)
          new_img[...,i]=img_
#      plt.figure('img')
#      plt.imshow(img[...,1000])      
#      plt.figure('img2')
#      plt.imshow(new_img[...,1000])
      img = new_img

  norm = img.sum(axis=0).sum(axis=0)
  img /= norm
  
  idx = np.where(wl <= 0.0)
  if idx:
    img[..., idx] *= 0.0
  
  return img
   
  
def PixelResponseFunction(opt, psf_shape, osf, delta, lx = 1.7*u.um, ipd = 0.0*u.um):
  '''
  Estimate the detector pixel response function with the prescription of 
  Barron et al., PASP, 119, 466-475 (2007).
  
  Parameters
  ----------
  psf_shape	: touple of scalars 
		  (ny, nx) defining the PSF size	
  osf		: scalar
		  number of samples in each resolving element. The 
		  final shape of the response function would be shape*osf
  delta 	: scalar
		  Phisical size of the detector pixel in microns
  lx		: scalar
		  diffusion length in microns
  ipd           : scalar
		  distance between two adjacent detector pixels 
		  in microns
		 
  Returns
  -------
  kernel	: 2D array
		  the kernel image
  kernel_delta  : scalar
                  the kernel sampling interval in microns
  '''
  if type(osf) != int: osf = np.int(osf)
    
#  lx = 0*u.um # top hat
  
  lx += 1e-8*u.um # to avoid problems if user pass lx=0
#==============================================================================
#  lx = 3.7*u.um # approximates Hardy et al
#==============================================================================
 
  lx = lx.to(delta.unit)
 
  jexosim_msg ("diffusion length in IPRF %s"%(lx), opt.diagnostics)
 
  
#==========FOR IMAGES ONLY====================================================================
#  osf*=33 # for image demo only 
#==============================================================================
  kernel = np.zeros( (psf_shape[0]*osf, psf_shape[1]*osf) )
  
#===========FOR IMAGES ONL===================================================================
#  kernel = np.zeros( (3*osf, 3*osf) )   # for image demo only
#==============================================================================
    
  kernel_delta = delta/osf
  yc, xc = np.array(kernel.shape) // 2
  yy = (np.arange(kernel.shape[0]) - yc) * kernel_delta 
  xx = (np.arange(kernel.shape[1]) - xc) * kernel_delta 
  mask_xx = np.where(np.abs(xx) > 0.5*(delta-ipd))
  mask_yy = np.where(np.abs(yy) > 0.5*(delta-ipd))
  xx, yy = np.meshgrid(xx, yy)
  

  
  kernel = np.arctan(np.tanh( 0.5*( 0.5*delta.value - xx.value)/lx.value )) - \
	   np.arctan(np.tanh( 0.5*(-0.5*delta.value - xx.value)/lx.value ))
	 	 
  kernel*= np.arctan(np.tanh( 0.5*( 0.5*delta.value - yy.value)/lx.value )) - \
  	   np.arctan(np.tanh( 0.5*(-0.5*delta.value - yy.value)/lx.value )) 

# for cross-talk (e.g if bell shaped function like Hardy) comment out these lines below.
  kernel[mask_yy, ...] = 0.0  
  kernel[..., mask_xx] = 0.0
#
#  # Normalise the kernel such that the pixel has QE=1
  kernel *= osf**2/kernel.sum()
  
 
  
#==============================================================================
# Below is for images only  
# 
#   
#  from mpl_toolkits.mplot3d import Axes3D
#  import matplotlib.pyplot as plt
#  from matplotlib import cm
#  from matplotlib.ticker import LinearLocator, FormatStrFormatter
#
#
#  fig = plt.figure('IPRF')
#  ax = fig.gca(projection='3d')
# 
#  surf = ax.plot_surface(xx/18e-6, yy/18e-6, kernel, cmap=cm.coolwarm,
#                       linewidth=0, antialiased=False)
#                       
#  ax.set_xlim(-1.5, 1.5)
#  ax.set_ylim(-1.5, 1.5)
#  ax.set_zlim(0, 1.0)
#  ax.set_xlabel('X axis (pixels)')
#  ax.set_ylabel('Y axis (pixels)')
#  ax.set_zlabel('Relative count per pixel')                     
# 
#  fig.colorbar(surf, shrink=0.5, aspect=5)
#
#  plt.show()  
#  
#  plt.figure('X-section through IPRF')
#  X,Y = np.unravel_index(kernel.argmax(), kernel.shape)
#  
#  import pandas as pd
#  qq = np.array(pd.read_csv('/Users/user1/Desktop/JWST_IRPRF_Hardy.csv'))
#  qq = qq[qq[:,0].argsort()]
# 
#  xx0 = np.linspace(3,8,1000)
# 
#  yy0  = np.interp(xx0,qq[:,0],qq[:,1])
#  idx = np.argwhere(yy0>=0)
#  yy0= yy0[idx]
#  xx0=xx0[idx]
#  idx = np.argmax(yy0)
#  xx0 =xx0-xx0[idx]
#  yy0 = yy0/yy0.max()
#  plt.plot(xx0+0.012,yy0, 'r-', linewidth=2, label='Hardy et al. 2014')
#  plt.plot(xx[0]/18e-6, kernel[X]/kernel[X].max(), 'b--',linewidth=2, label = 'JexoSim')
#  plt.grid(True)
#  plt.xlabel('Distance (pixels)')
#  plt.ylabel('Relative response')
#  
#  ax = plt.gca() 
#
#
#  legend = ax.legend(loc='upper right', shadow=True)
#  frame = legend.get_frame()
#  frame.set_facecolor('0.90')
#  for label in legend.get_texts():
##    label.set_fontsize('medium')
#    label.set_fontsize(22)
#  for label in legend.get_lines():
#    label.set_linewidth(1.5)  # the legend line width
#  for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
#             ax.get_xticklabels() + ax.get_yticklabels()):
#    item.set_fontsize(22)
#    
#  xxx
 
  
#==============================================================================
 
  kernel = np.roll(kernel, -xc, axis=1)
  kernel = np.roll(kernel, -yc, axis=0)
  
  return kernel, kernel_delta


def pointing_jitter(opt):
    
  jitter_file = opt.jitter_psd_file 
  total_observing_time = opt.total_observing_time
  frame_time = opt.frame_time
  rms = opt.pointing_model.pointing_rms.val
    
  ''' Estimate pointing jitter timeline
  
  Parameters
  ----------
  jitter_file: string
	       filename containing CSV columns with 
	       frequency [Hz], Yaw PSD [deg**2/Hz], Pitch [deg**2/Hz]
	       If only two columns given, then it is assumed that 
	       the second column is the PSD of radial displacements
  totoal_observing_time: scalar
      total observing time in units of time
  frame_time: scalar
      detector frame time in units of time
  rms: scalar
      renormalisation rms in units of angle
      
  Returns
  -------
  yaw_jit: jitter timeline in units of degrees
  pitch_jit: jitter rimeline in units of degrees
  osf: number of additional samples in each frame_time needed to capture
       the jitter spectral information
  '''
  
  data = np.genfromtxt(jitter_file, delimiter=',')
  psd_freq = data[..., 0]

  if data.shape[1] > 2:
    psd_yaw = data[..., 1]
    psd_pitch = data[..., 2]
  else:
    psd_yaw = data[..., 1]/2
    psd_pitch = psd_yaw
  
#  psd_yaw = psd_yaw[1:]
#  psd_pitch = psd_pitch[1:] 
#  psd_freq = psd_freq[1:]
  
#  
#  if opt.Inc_PSD ==1:    
#          psd_yaw = psd_yaw * 1.52
#          psd_pitch = psd_pitch * 1.52  
          
#  if np.sum(opt.psd) >0 :
#      psd_freq = np.linspace(1/(10*3600.), 10, 100)
#      psd_yaw = np.ones(len(psd_freq))*opt.psd
#      psd_pitch = np.ones(len(psd_freq))*opt.psd
#      

  # each frame needs to be split such that jitter is Nyquis sampled
  jitter_sps = 2.0*psd_freq.max()/u.s
  
  osf = (np.ceil((frame_time).to(u.s) * jitter_sps).take(0).astype(np.int)).value

  if osf < 1: osf = 1
      
  jexosim_msg ("Frame OSF (number of jitter frames per NDR %s"%(osf) , opt.diagnostics)

  number_of_samples_ = np.int(osf*np.ceil(total_observing_time/frame_time))  + 100
  N0 = number_of_samples_ 
 

  number_of_samples = 2**(np.ceil(np.log2(number_of_samples_)))
  

  number_of_samples = int( (number_of_samples /2)+1  )
#  number_of_samples += 100

## for same random sim 
#  ind= int(np.log10(number_of_samples))
#  rnd = 10**ind
#  number_of_samples = rnd + np.ceil(number_of_samples/ rnd) * rnd

  freq = np.linspace(0.0, 0.5* osf/ (frame_time).to(u.s).value, number_of_samples)


  
  npsd_yaw   = 1.0e-30+np.interp(freq, psd_freq, psd_yaw, left=0.0, right=0.0)
  npsd_pitch = 1.0e-30+np.interp(freq, psd_freq, psd_pitch, left=0.0, right=0.0)

 
   
  jexosim_msg ("interpolation of PSD done", opt.diagnostics)
#  import matplotlib.pyplot as plt
#  plt.figure(33)
#  plt.plot(psd_freq,psd_yaw, 'rx-')
#  plt.plot(freq,npsd_yaw,'bx-')


  npsd_yaw    = np.sqrt(npsd_yaw   * np.gradient(freq))
  npsd_pitch  = np.sqrt(npsd_pitch * np.gradient(freq))

 
  yaw_jit_re   = np.random.normal(scale=npsd_yaw/2.0)
  yaw_jit_im   = np.random.normal(scale=npsd_yaw/2.0)
  pitch_jit_re = np.random.normal(scale=npsd_pitch/2.0)
  pitch_jit_im = np.random.normal(scale=npsd_pitch/2.0)
  
  pitch_jit_im[0] = pitch_jit_im[-1] = 0.0
  yaw_jit_im[0]   = yaw_jit_im[-1]   = 0.0


  norm = 2*(number_of_samples-1)

  
  jexosim_msg ("starting irfft" , opt.diagnostics)
  yaw_jit = norm*np.fft.irfft(yaw_jit_re + 1j * yaw_jit_im)*u.deg
  pitch_jit = norm*np.fft.irfft(pitch_jit_re + 1j * pitch_jit_im)*u.deg
  jexosim_msg ("completed.....", opt.diagnostics)

  if opt.pointing_model.adjust_rms.val ==1:
    norm = (rms**2/(yaw_jit.var()+ pitch_jit.var())).simplified
    yaw_jit *= np.sqrt(norm)
    pitch_jit *= np.sqrt(norm)


   
  if len(yaw_jit) > N0:
      yaw_jit = yaw_jit[0:N0]
      pitch_jit = pitch_jit[0:N0]  
            
  
  
  jexosim_msg ("jitter RMS in mas %s %s"%(np.std (yaw_jit)*3600*1000, np.std(pitch_jit)*3600*1000) , opt.diagnostics)
    
  return yaw_jit, pitch_jit, osf


def oversample(fp, ad_osf):
    
    xin = np.linspace(0,fp.shape[1]-1,fp.shape[1])
    yin = np.linspace(0,fp.shape[0]-1,fp.shape[0])
    x_step =  abs(xin[1]) - abs(xin[0])
    y_step =  abs(yin[1]) - abs(yin[0])
    
    # calculates the new step sizes for new grid
    x_step_new = np.float(x_step/ad_osf)
    y_step_new = np.float(y_step/ad_osf)
    
    # new grid must start with an exact offset to produce correct number of new points
    x_start = -x_step_new * np.float((ad_osf-1)/2)
    y_start = -y_step_new * np.float((ad_osf-1)/2)
    
    # new grid points- with correct start, end and spacing
    xout = np.arange(x_start, x_start + x_step_new*fp.shape[1]*ad_osf, x_step_new)
    yout = np.arange(y_start, y_start + y_step_new*fp.shape[0]*ad_osf, y_step_new)
    
    # interpolate fp onto new grid
    fn = interpolate.RectBivariateSpline(yin,xin, fp)
    new_fp = fn(yout,xout)
    
    return new_fp


def write_record(opt, path, lab):
    


    textfile = '%s/%s.txt'%(path,lab)
    file = open(textfile,'w')
    file.write('Planet:  %s'%(opt.planet.planet.name))
    file.write('\nChannel:  %s'%(opt.channel.name))
    file.write('\n\nNoise option:  %s'%(opt.noise_tag))
    
    file.write('\n ')
    file.write('\nObservation feasible?:  %s'%(opt.observation_feasibility))
 
    file.write('\n ')
    file.write('\nUse saturation time?:  %s'%(opt.channel.detector_readout.use_sat.val))
    file.write('\nSat time (to designated fraction of full well):  %s sec'%(opt.sat_time))
    file.write('\nt_f:  %s sec'%(opt.channel.detector_readout.t_f.val.value.item()))
    file.write('\nt_g:  %s sec'%(opt.channel.detector_readout.t_g.val.value.item()))
    file.write('\nt_sim:  %s sec'%(opt.channel.detector_readout.t_sim.val.value.item()))
    file.write('\nsubarray  :  %s'%(opt.subarray))

    if opt.observation_feasibility ==1:
        file.write('\nt_int:  %s sec'%(opt.t_int.value.item()))
        file.write('\nt_cycle:  %s sec'%(opt.exposure_time.value.item()))

        file.write('\nprojected multiaccum (n groups):  %s'%(opt.projected_multiaccum))
        file.write('\neffective multiaccum (n groups):  %s'%(opt.effective_multiaccum))
        file.write('\nnumber of NDRs simulated:  %s'%(opt.n_ndr) )
        file.write('\nnumber of integration cycles:  %s'%(opt.n_exp) )
        
        file.write('\n ')
        file.write('\nApFactor:  %s'%(opt.pipeline.ApFactor) )
        file.write('\nAperture shape:  %s'%(opt.channel.data_pipeline.ApShape.val) )
        file.write('\nSpectral binning:  %s '%(opt.channel.data_pipeline.binning.val) )
        if opt.channel.data_pipeline.binning.val == 'R-bin':
                   file.write('\nBinned R power:  %s '%(opt.channel.data_pipeline.R.val) )
        else:
                  file.write('\nBinned R power:  %s '%(opt.channel.data_pipeline.bin_size.val) )
        
        
    file.close()

