
import numpy   as np
from astropy.io import fits
from astropy import units as u
from jexosim.classes import sed
from astropy import constants as const
import os, sys
from jexosim.lib.jexosim_lib import jexosim_msg, planck, jexosim_plot
import jexosim

class Star():

    def __init__(self, opt):
      
        self.opt = opt
        self.star_temperature = opt.exosystem.star.T 
        self.star_logg = opt.exosystem.star.logg 
        self.star_f_h = opt.exosystem.star.Z # currently only 0 can be chosen from files
        
        if opt.exosystem_params.star_spectrum_model.val =='complex':
            self.complex_model()
            jexosim_msg ("complex star spectrum chosen from database", 1)           
        elif opt.exosystem_params.star_spectrum_model.val == 'simple':
            self.simple_model() 
            jexosim_msg ("simple star spectrum chosen", 1)
        elif opt.exosystem_params.star_spectrum_model.val =='file':
            self.file_model()  
            jexosim_msg ("filed star spectrum chosen", 1)
        else:
            jexosim_msg('Error1 star class: no compatible entry for star spectrum_model', 1)
            sys.exit()
           
        stellar_sed_pre_norm =  self.stellar_sed*1 
           
        if self.opt.exosystem_params.star_spectrum_mag_norm.val == 1:
            self.stellar_sed  = self.useTelFlux(self.stellar_wl, self.stellar_sed)
        elif  self.opt.exosystem_params.star_spectrum_mag_norm.val == 2:        
            self.stellar_sed  *=  ((self.opt.exosystem.star.R).to(u.m)/(self.opt.exosystem.star.d).to(u.m))**2 		      # [W/m^2/mu]
        elif  self.opt.exosystem_params.star_spectrum_mag_norm.val == 0:    
            pass
        
        jexosim_msg("star check 2: %s"%self.stellar_sed.max(), self.opt.diagnostics)
        self.sed = sed.Sed(self.stellar_wl, self.stellar_sed)    
        self.sed_pre_norm = sed.Sed(self.stellar_wl, stellar_sed_pre_norm)   
     
        
        
        import matplotlib.pyplot as plt 
        plt.figure('test star spectrum 1')
        plt.plot(self.stellar_wl, self.stellar_sed)
        plt.figure('test star spectrum pre-norm')
        plt.plot(self.stellar_wl,  stellar_sed_pre_norm)
 
 
    
    def complex_model(self):
    
        jexosim_msg ("Star temperature:  %s"%(self.opt.exosystem.star.T), self.opt.diagnostics)
        jexosim_msg ("Star name %s, dist %s, radius %s"%(self.opt.exosystem.star.name, self.opt.exosystem.star.d, self.opt.exosystem.star.R), self.opt.diagnostics)  
        jexosim_path =  os.path.dirname((os.path.dirname(jexosim.__file__)))
        databases_dir = '%s/archive'%(jexosim_path)   
        cond=0
        for root, dirs, files in os.walk(databases_dir):
            for dirc in dirs:
                if 'BT-Settl' in dirc:
                    dirc_name = dirc
                    cond=1
                    break
        if cond==0:
            jexosim_msg('Error Star 1: database not found')   
            sys.exit()
        self.sed_folder = '%s/%s'%(databases_dir, dirc_name)
           
        # ph_wl, ph_sed = self.read_phoenix_spectrum()  
        self.stellar_wl, self.stellar_sed = self.read_phoenix_spectrum_interp()  
  
        # import matplotlib.pyplot as plt
        # plt.figure('test 1')
        # plt.plot(ph_wl, ph_sed)
        
        # plt.figure('test 2')
        # plt.plot(ph_wl, np.gradient(ph_wl))

        # plt.figure('test 3 - R power')
        # plt.plot(ph_wl, ph_wl/np.gradient(ph_wl))  
     
  
    def simple_model(self):
   
        jexosim_msg ("Using black body spectrum for star...", self.opt.diagnostics)
        self.stellar_wl = np.linspace(0.4,18.0,int(1e6))*u.um 
        self.stellar_sed =  np.pi*planck(self.stellar_wl, self.star_temperature) * u.sr
          
    def file_model(self):
        
        filename = self.opt.exosystem_params.star_spectrum_file.val
        # if the file is in the planet_spectra folder can be IDed only with filename not full path
        if '/' in  filename:
            pass
        else:
            filename = '%s/jexosim/data/star_spectra/%s'%(self.opt.jexosim_path, filename)
        
        jexosim_msg('star spectrum from file: %s'%(filename),1)  
        
        try:
            aa = np.loadtxt(filename)
        except IOError:
            jexosim_msg("Error6  planet class: No spectrum file found",  1) 
        # if provided in m... convert to um
        if aa[:,0][0]< 1e-4: 
            self.stellar_wl=  aa[:,0]*1e6*u.um 
        else:
            self.stellar_wl=  aa[:,0]*u.um 
        self.stellar_sed = aa[:,1]*u.W/(u.m)**2/u.um
   
   
 
    
    def useTelFlux(self, wl, sed):
  
          K_mag = self.opt.model_exosystem.K_mag.val
          J_mag = self.opt.model_exosystem.J_mag.val
          mag_band_list =['J', 'K']
          
          print ('hello', K_mag)
          print ('hello', J_mag)
          if K_mag == '':
              mag_band_list =['J', 'J']
          if J_mag == '':
              mag_band_list =['K', 'K']
          if J_mag == '' and K_mag == '':
              jexosim_msg('Star class error: no J or K magnitude entered')
              sys.exit()
              
          sed_list =[] # take an average of J and K normalisations
          
          if 'star_test_mag' in self.opt.input_params:
              if self.opt.input_params['star_test_mag'] == 8.0:  # this is an internal test 
                  mag_band_list =['J', 'J']
                  J_mag = 8.0
              
    
          jexosim_msg ("normalizing stellar spectrum",  self.opt.diagnostics)
            
          wl = wl.value
          for mag_band in mag_band_list:
                    
              if mag_band == 'J':
                  # wav = 1.235
                  # Jy = 1594       
                  wav = 1.25
                  Jy = 1603 
                  mag = J_mag
              if mag_band == 'K':
                  wav = 2.22
                  Jy = 667   
                  mag = K_mag
              F_0 = const.c.value*1e6*1e-26*(Jy)/wav**2 
              F_x = F_0*10**(-mag/2.5)               
              xx=10000
              box = np.ones(xx)/xx
              xx2 = 1000
              box2 = np.ones(xx2)/xx2
              sed0 = np.convolve(sed,box,'same')  
              idx = np.argwhere(wl>=wav)[0].item()
              F_s = sed0[idx]    
              scale = F_x/F_s
              sed1 = sed*scale    
              sed_list.append(sed1)
              sed1 = sed1*u.W/u.m**2/u.um
              import matplotlib.pyplot as plt
              plt.figure('sed normalization')
              plt.plot(wl, np.convolve(sed1,box2,'same')  , label = mag_band)
              
        
          sed_final = (np.array(sed_list[0]) + np.array(sed_list[1])) /2.0
          plt.plot(wl, np.convolve(sed_final,box2,'same'), label='average of J and K norm' )
                    
          plt.plot(wl, sed0*(self.opt.exosystem.star.R.to(u.m)/\
                             self.opt.exosystem.star.d.to(u.m))**2, label='*(Rs/D)^2' )		      # [W/m^2/mu]., 'r-')
          plt.legend()
      
          sed_final = sed_final*u.W/u.m**2/u.um
          return sed_final
    

    def read_phoenix_spectrum(self):
        jexosim_msg ("Star... star_temperature %s, star_logg %s, star_f_h %s"%(self.star_temperature, self.star_logg, self.star_f_h) , self.opt.diagnostics)
        if self.star_temperature.value >= 2400:
            t0 = np.round(self.star_temperature,-2)/100.
        else:
            t0= np.round(np.round(self.star_temperature,-1)/100./0.5)*0.5
        logg0 =  np.round(np.round(self.star_logg,1)/0.5)*0.5    
        cond=0     
        for root, dirs, files in os.walk(self.sed_folder):
            for filename in files:
                if filename == 'lte0%s-%s-0.0a+0.0.BT-Settl.spec.fits'%(t0.value, logg0):
                    ph_file = '%s/%s'%(self.sed_folder, filename)
                    cond = 1
                elif filename == 'lte0%s-%s-0.0a+0.0.BT-Settl.spec.fits.gz'%(t0.value, logg0):
                    ph_file = '%s/%s'%(self.sed_folder, filename)
                    jexosim_msg ("Star file used:  %s"%(ph_file) , self.opt.diagnostics)
                    cond = 1
        if cond==0:
            jexosim_msg ("Error Star 2:  no star file found", 1)
            sys.exit()                
        with fits.open(ph_file) as hdu:
            wl = u.Quantity(hdu[1].data.field('Wavelength'),
                         hdu[1].header['TUNIT1']).astype(np.float64)
            sed = u.Quantity(hdu[1].data.field('Flux'), u.W/u.m**2/u.um).astype(np.float64)
            if hdu[1].header['TUNIT2'] != 'W / (m2 um)': print ('Exception')                        
         #remove duplicates
            idx = np.nonzero(np.diff(wl))
            wl = wl[idx]
            sed = sed[idx]
            hdu.close()
        jexosim_msg("star check 1: %s"%sed.max(), self.opt.diagnostics)
        return wl, sed
    
 
    def read_phoenix_spectrum_interp(self):

        jexosim_msg ("Star... star_temperature %s, star_logg %s, star_f_h %s"%(self.star_temperature, self.star_logg, self.star_f_h) , self.opt.diagnostics)
       
        if self.star_temperature.value >= 2400:
            t0 = np.round(self.star_temperature,-2)/100.
        else:
            t0= np.round(np.round(self.star_temperature,-1)/100./0.5)*0.5
            
        t0 = t0.value
        
        if t0*100 != self.star_temperature.value:
            cond_1 = 1
            if t0*100 > self.star_temperature.value:
                    if self.star_temperature.value >= 2400:
                        t1 = t0-1
                    else:
                        t1 = t0-0.5
            elif t0*100 < self.star_temperature.value:
                    if self.star_temperature.value >= 2400:
                        t1= t0+1
                    else:
                        t1 = t0+0.5
            print (t0,t1, self.star_temperature)
        else:
            cond_1 = 0 # do not interp for temp
            
                        
        logg0 =  np.round(np.round(self.star_logg,1)/0.5)*0.5
        if logg0 <= 2.5:
            logg0 = 2.5
            cond_2 = 0 # do not interp for logg
        elif logg0 >= 5.5:
            logg0 = 5.5
            cond_2 = 0
        elif logg0 == self.star_logg:
            cond_2 = 0 
        else:
            cond_2 = 1
            if logg0 < self.star_logg:
                logg1 = logg0 + 0.5 
            if logg0 > self.star_logg:
                logg1 = logg0 - 0.5
                                         
        if cond_1 == 1 and cond_2 == 0: 
            
            jexosim_msg ("interpolating spectra by temperature only", self.opt.diagnostics)
      
            wav_t0_logg0, flux_t0_logg0 = self.open_Phoenix(t0,logg0, self.star_f_h, self.sed_folder)
            wav_t1_logg0, flux_t1_logg0 = self.open_Phoenix(t1,logg0, self.star_f_h, self.sed_folder)
            sed_t0_logg0 = sed.Sed(wav_t0_logg0, flux_t0_logg0)
            sed_t1_logg0 = sed.Sed(wav_t1_logg0, flux_t1_logg0)
            # bin to same wav grid
            sed_t1_logg0.rebin(wav_t0_logg0)  
            wt0 = 1 - abs(t0*100 - self.star_temperature.value)/abs(t0*100 - t1*100)
            wt1 = 1 - abs(self.star_temperature.value - t1*100)/abs(t0*100 - t1*100)
            sed_final = wt0*sed_t0_logg0.sed + wt1*sed_t1_logg0.sed
            
            star_final_sed = sed_final
            star_final_wl = sed_t0_logg0.wl
            
            jexosim_plot('temperature interpolation', self.opt.diagnostics, xdata = wav_t0_logg0, ydata = flux_t0_logg0, marker ='r-')
            jexosim_plot('temperature interpolation', self.opt.diagnostics, xdata = wav_t1_logg0, ydata = flux_t1_logg0, marker ='b-')
            jexosim_plot('temperature interpolation', self.opt.diagnostics, xdata = star_final_wl, ydata = star_final_sed, marker ='g-')
    
          
        if cond_1 == 1 and cond_2 == 1:
            
            jexosim_msg ("interpolating spectra by temperature and logg", self.opt.diagnostics)
       
            #interp temp at logg0
            wav_t0_logg0, flux_t0_logg0 = self.open_Phoenix(t0,logg0, self.star_f_h, self.sed_folder)
            wav_t1_logg0, flux_t1_logg0 = self.open_Phoenix(t1,logg0, self.star_f_h, self.sed_folder)
            sed_t0_logg0 = sed.Sed(wav_t0_logg0, flux_t0_logg0)
            sed_t1_logg0 = sed.Sed(wav_t1_logg0, flux_t1_logg0)
            # bin to same wav grid
            sed_t1_logg0.rebin(wav_t0_logg0)  
            wt0 = 1 - abs(t0*100 - self.star_temperature.value)/abs(t0*100 - t1*100)
            wt1 = 1 - abs(self.star_temperature.value - t1*100)/abs(t0*100 - t1*100)
            sed_ = wt0*sed_t0_logg0.sed + wt1*sed_t1_logg0.sed
            sed_logg0 = sed.Sed(wav_t0_logg0, sed_)
            
            print (t0,t1)
            print (wt0,wt1)
             
            box = 1000
            bbox = np.ones(box)/box
            jexosim_plot('temperature interpolation logg0', self.opt.diagnostics, xdata = wav_t0_logg0, \
                         ydata = np.convolve(flux_t0_logg0, bbox,'same'), marker ='r-')
            jexosim_plot('temperature interpolation logg0', self.opt.diagnostics, xdata = wav_t1_logg0, \
                         ydata = np.convolve(flux_t1_logg0, bbox,'same'), marker ='b-')
            jexosim_plot('temperature interpolation logg0', self.opt.diagnostics, xdata = sed_logg0.wl, \
                         ydata = np.convolve(sed_logg0.sed, bbox,'same'), marker ='g-')
            
    
            #interp temp at logg1
            wav_t0_logg1, flux_t0_logg1 = self.open_Phoenix(t0,logg1, self.star_f_h, self.sed_folder)
            wav_t1_logg1, flux_t1_logg1 = self.open_Phoenix(t1,logg1, self.star_f_h, self.sed_folder)
            sed_t0_logg1 = sed.Sed(wav_t0_logg1, flux_t0_logg1)
            sed_t1_logg1 = sed.Sed(wav_t1_logg1, flux_t1_logg1)
            # bin to same wav grid
            sed_t1_logg1.rebin(wav_t0_logg1)  
            wt0 = 1 - abs(t0*100 - self.star_temperature.value)/abs(t0*100 - t1*100)
            wt1 = 1 - abs(self.star_temperature.value - t1*100)/abs(t0*100 - t1*100)
            sed_ = wt0*sed_t0_logg1.sed + wt1*sed_t1_logg1.sed
            sed_logg1 = sed.Sed(wav_t0_logg1, sed_)
            
            box = 100
            bbox = np.ones(box)/box
            jexosim_plot('temperature interpolation logg1', self.opt.diagnostics, xdata = wav_t0_logg1, \
                         ydata = np.convolve(flux_t0_logg1, bbox,'same'), marker ='r-')
            jexosim_plot('temperature interpolation logg1', self.opt.diagnostics, xdata = wav_t1_logg1, \
                         ydata = np.convolve(flux_t1_logg1, bbox,'same'), marker ='b-')
            jexosim_plot('temperature interpolation logg1', self.opt.diagnostics, xdata = sed_logg1.wl, \
                         ydata = np.convolve(sed_logg1.sed, bbox,'same'), marker ='g-')
            
      
            # now interp logg
            sed_logg1.rebin(sed_logg0.wl)
            wt0 = 1 - abs(logg0 - self.star_logg)/abs(logg0 - logg1)
            wt1 = 1 - abs(self.star_logg - logg1)/abs(logg0 - logg1)
            print (wt0,wt1)
            print (logg0,logg1)
        
             
            sed_final = wt0*sed_logg0.sed + wt1*sed_logg1.sed
       
            star_final_sed = sed_final
            star_final_wl = sed_logg0.wl
            
            box = 100
            bbox = np.ones(box)/box
            jexosim_plot('interpolation between logg', self.opt.diagnostics, xdata = sed_logg0.wl, \
                         ydata = np.convolve(sed_logg0.sed, bbox,'same'), marker ='r-')
            jexosim_plot('interpolation between logg', self.opt.diagnostics, xdata = sed_logg1.wl, \
                         ydata = np.convolve(sed_logg1.sed, bbox,'same'), marker ='b-')
            jexosim_plot('interpolation between logg', self.opt.diagnostics, xdata = star_final_wl, \
                         ydata = np.convolve(star_final_sed, bbox,'same'), marker ='g-') 
      
                
        if cond_1 == 0 and cond_2 == 1:
            
            jexosim_msg ("interpolating spectra by logg only", self.opt.diagnostics)
       
            #interp logg at t0
            wav_t0_logg0, flux_t0_logg0 = self.open_Phoenix(t0,logg0, self.star_f_h, self.sed_folder)
            wav_t0_logg1, flux_t0_logg1 = self.open_Phoenix(t0,logg1, self.star_f_h, self.sed_folder)
             
            sed_t0_logg0 = sed.Sed(wav_t0_logg0, flux_t0_logg0)
            sed_t0_logg1 = sed.Sed(wav_t0_logg1, flux_t0_logg1)
            # bin to same wav grid
            sed_t0_logg1.rebin(wav_t0_logg0) 
    
            wt0 = 1 - abs(logg0 - self.star_logg)/abs(logg0 - logg1)
            wt1 = 1 - abs(self.star_logg - logg1)/abs(logg0 - logg1)
            sed_final = wt0*sed_logg0.sed + wt1*sed_logg1.sed
            
            star_final_sed = sed_final
            star_final_wl = sed_t0_logg0.wl  
            
            box = 100
            bbox = np.ones(box)/box
            jexosim_plot('interpolation between logg', self.opt.diagnostics, xdata = sed_t0_logg0.wl, \
                         ydata = np.convolve(sed_t0_logg0.sed, bbox,'same'), marker ='r-')
            jexosim_plot('interpolation between logg', self.opt.diagnostics, xdata = sed_t0_logg1.wl, \
                         ydata = np.convolve(sed_t0_logg1.sed, bbox,'same'), marker ='b-')
            jexosim_plot('interpolation between logg', self.opt.diagnostics, xdata = star_final_wl, \
                         ydata = np.convolve(star_final_sed, bbox,'same'), marker ='g-') 
                
        if cond_1 == 0 and cond_2 == 0:
            
            star_final_wl, star_final_sed = self.open_Phoenix(t0,logg0, self.star_f_h, self.sed_folder)
  
        return  star_final_wl, star_final_sed  
            
 
    def open_Phoenix(self, t0, logg0, Z, sed_folder):
    
        cond = 0
        for root, dirs, files in os.walk(sed_folder):
            for filename in files:
                if filename == 'lte0%s-%s-0.0a+0.0.BT-Settl.spec.fits'%(t0, logg0):
                    ph_file = '%s/%s'%(sed_folder, filename)
                    cond = 1
                elif filename == 'lte0%s-%s-0.0a+0.0.BT-Settl.spec.fits.gz'%(t0, logg0):
                    ph_file = '%s/%s'%(sed_folder, filename)
                    jexosim_msg ("Star file used:  %s"%(ph_file) , self.opt.diagnostics)
                    cond = 1
        if cond==0:
            jexosim_msg ("Error Star 2:  no star file found", 1)
            sys.exit()    
        else:
            with fits.open(ph_file) as hdu:
                wl = u.Quantity(hdu[1].data.field('Wavelength'),
                             hdu[1].header['TUNIT1']).astype(np.float64)
                sed = u.Quantity(hdu[1].data.field('Flux'), u.W/u.m**2/u.um).astype(np.float64)
                if hdu[1].header['TUNIT2'] != 'W / (m2 um)': print ('Exception')
                                 
             #remove duplicates
                idx = np.nonzero(np.diff(wl))
                wl = wl[idx]
                sed = sed[idx]
                hdu.close()
     
        return wl, sed
     
     