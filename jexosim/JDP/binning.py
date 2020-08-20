"""
JexoSim 
2.0
JDP binning module
v1.0

"""

import numpy as np
from scipy import interpolate, optimize
import matplotlib.pyplot as plt
from scipy import interpolate
import sys
import pytransit
# import pylightcurve
from scipy.optimize import minimize
from astropy import units as u
from pytransit import QuadraticModel, SwiftModel
from jexosim.lib.jexosim_lib import jexosim_msg, jexosim_plot





#==============================================================================
# Processing of transit light curves: fitting and extraction of transit depths
#==============================================================================


class processLC():
    def __init__(self, LC, binnedWav, binnedGamma, opt):
        self.opt = opt
        
        self.timegrid = opt.z_params[0]
        self.t0 = opt.z_params[1]
        self.per = opt.z_params[2]
        self.ars = opt.z_params[3]
        self.inc = opt.z_params[4]
        self.ecc = opt.z_params[5]
        self.omega = opt.z_params[6] 
 
#        self.opt.channel.data_pipeline.useReduced.val =1
     
        try: 
            self.opt.LDClaw
        except AttributeError:
            self.opt.LDClaw ='quad'
            
        if self.opt.LDClaw !='claret4':       
           self.tm = QuadraticModel(interpolate=False)
        else:
           self.tm = SwiftModel(interpolate=False)          
        self.tm.set_data(self.timegrid)
                     

        self.LC = LC
        self.binnedWav = binnedWav
        self.binnedGamma = binnedGamma
        self.transitDepths = []
        self.transitDepths_err =[]
        self.model_gamma1 = []
        self.model_gamma2 = []
        self.model_gamma3 = []
        self.model_gamma4 = []    
        self.model_f = []
        self.finalWav = []
        self.fitModelLC()
        
        
        if self.opt.channel.data_pipeline.useReduced.val == 1:
            self.binnedWav = self.finalWav
          
 
    def fitModelLC(self):
        
        nWLs = self.LC.shape[1]  # how many steps in the loop
    
        # Progress Bar setup:
        ProgMax = 100    # number of dots in progress bar
        if nWLs<ProgMax:   ProgMax = nWLs   # if less than 20 points in scan, shorten bar
        print ("|" +    ProgMax*"-"    + "|     Fitting model light curves")
        sys.stdout.write('|'); sys.stdout.flush();  # jexosim_msg start of progress bar
        nProg = 0   # fraction of progress   

        step = 1
        if self.opt.channel.data_pipeline.useReduced.val == 1:
            jexosim_msg ("using reduced number of R-bins for speed..." , self.opt.diagnostics)
            step = 5        
        
             
        for i in range(0, self.LC.shape[1], step):
            self.finalWav.append(self.binnedWav[i])
            
            if ( i >= nProg*nWLs/ProgMax ):
                    sys.stdout.write('*'); sys.stdout.flush();  
                    nProg = nProg+1
            if ( i >= nWLs-1 ):
                    sys.stdout.write('|     done  \n'); sys.stdout.flush(); 

            self.dataLC = self.LC[:,i]
            
            #fit light curve
            self.final_fit = self.light_curve_fit(i)
            
            p = self.final_fit[0]**2 #to give (Rp/Rs)^2
            f = self.final_fit[1]
            # p_err = self.final_fit[-1]
            self.transitDepths.append(p)  
            # self.transitDepths_err.append(p_err)
            self.model_f.append(f)   
            
            gamma1 = self.final_fit[2]
            gamma2 = self.final_fit[3]
            self.model_gamma1.append(gamma1)
            self.model_gamma2.append(gamma2)
            
            if self.opt.LDClaw == 'claret4':
                            gamma3 = self.final_fit[4]
                            gamma4 = self.final_fit[5]
                            self.model_gamma3.append(gamma3)
                            self.model_gamma4.append(gamma4)
                
            jexosim_msg ('%s %s %s %s'%(i, len(self.transitDepths), p, self.binnedWav[i]),  self.opt.diagnostics)      

    
        
    def getModelLC_integrated(self, k, gamma):    
    
        modelLC_ = self.tm.evaluate(k=k, ldc=gamma, t0=self.t0, p=self.per, a=self.ars, i=self.inc, e=self.ecc, w=self.omega)   

        # now 'integrate' to match simulation
        lc = np.hstack((modelLC_, [0]))
        idx = np.hstack(([0],np.cumsum(self.opt.frames_per_ndr).astype(int)))   
        lc0 = np.add.reduceat(lc,idx)[:-1] /self.opt.frames_per_ndr
        lc0[lc0==np.inf] = 0
        # now 'do CDS' to match simulation
        lc_ndr_zero = lc0[0: len(lc0) : self.opt.effective_multiaccum]
        lc_ndr_final = lc0[self.opt.effective_multiaccum-1: len(lc0) : self.opt.effective_multiaccum]
        modelLC = lc_ndr_final - lc_ndr_zero
        
        return modelLC
        
    def getModelLC_instant(self, k, gamma):  

        
        modelLC_ = self.tm.evaluate(k=k, ldc=gamma, t0=self.t0, p=self.per, a=self.ars, i=self.inc, e=self.ecc, w=self.omega)   

        # now 'do CDS' to match simulation

        if self.opt.effective_multiaccum == 2:       
            lc0 = modelLC_
            lc0[lc0==np.inf] = 0
            modelLC =  modelLC_[1::2]
            
        else: 

            lc0 = modelLC_*1
            lc0[lc0==np.inf] = 0            
            
            n_exp = len(lc0)/self.opt.effective_multiaccum
            a = np.cumsum(lc0.reshape(n_exp, self.opt.effective_multiaccum), axis=1)
            t_ndr = self.opt.ndr_end_time[0:self.opt.effective_multiaccum]
            Sx = t_ndr.std()
            Sy = a.std(axis=1)
            R = (len(t_ndr)*(t_ndr*a).sum(axis=1) - t_ndr.sum()*a.sum(axis=1))  / \
            (np.sqrt( (len(t_ndr)*np.sum(t_ndr**2) - (np.sum(t_ndr))**2) * ( len(t_ndr)*(a**2).sum(axis=1) - (a.sum(axis=1))**2) ))  
            m = R*Sy/Sx
            modelLC   = m/m[0]          
                     
        return modelLC    
        
    def chi_sq (self, X):
        p = X[0]
        F = X[1]
        g1 = X[2]
        g2 = X[3]
      
        
        if self.opt.timeline.apply_lo_dens_LC.val == 1:
            model = self.getModelLC_instant(p, [g1,g2])* F 
        else:
            model = self.getModelLC_integrated(p, [g1,g2])* F  

        return np.sum(((model-self.dataLC))**2)
    
    def chi_sq_no_gamma (self, X, g1, g2):
        p = X[0]
        F = X[1]      
        
        if self.opt.timeline.apply_lo_dens_LC.val == 1:
            model = self.getModelLC_instant(p, [g1,g2])* F
        else:
            model = self.getModelLC_integrated(p, [g1,g2])* F  
        
        return np.sum(((model-self.dataLC))**2)
                
    def chi_sq_4 (self,X):
        p = X[0]
        F = X[1]
        g1 = X[2]
        g2 = X[3]      
        g3 = X[4]
        g4 = X[5]   
        
        if self.opt.timeline.apply_lo_dens_LC.val == 1:
            model = self.getModelLC_instant(p, [g1,g2, g3, g4])* F 
        else:
            model = self.getModelLC_integrated(p, [g1,g2, g3, g4])* F  

        return np.sum(((model-self.dataLC))**2)        
        
    def chi_sq_no_gamma_4 (self, X, g1, g2, g3, g4):
        p = X[0]
        F = X[1]    
        
        if self.opt.timeline.apply_lo_dens_LC.val == 1:
            model = self.getModelLC_instant(p, [g1,g2, g3, g4])* F 
        else:
            model = self.getModelLC_integrated(p, [g1,g2, g3, g4])* F  

        return np.sum(((model-self.dataLC))**2)        
        
        
    def light_curve_fit(self, i): 
        
        ex = self.dataLC
        
        err = np.std(np.hstack((ex[0:np.int(0.2*len(ex))], ex[np.int(0.8*len(ex)):])))
        oot_est = np.mean(np.hstack((ex[0:np.int(0.2*len(ex))], ex[np.int(0.8*len(ex)):])))
        it_est = np.mean(ex  [  int(len(ex)/2 -len(ex)/8)   :  int(len(ex)/2 +len(ex)/8)   ]     )
        cr_est = (oot_est - it_est ) / oot_est
        p_est = cr_est**0.5
        if err == 0:
            err = 1e-8      
         
        if self.opt.channel.data_pipeline.fit_gamma.val == 1: 
            if self.opt.LDClaw=='claret4':
                fit_init = [p_est, oot_est, 0.0, 0.0, 0.0, 0.0]
                fit  = minimize(self.chi_sq_4, fit_init, args=(), method='Nelder-Mead', jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None) 
                final_fit = [fit['x'][0], fit['x'][1], fit['x'][2], fit['x'][3],  fit['x'][4], fit['x'][5]]  
            else:      
                fit_init = [p_est, oot_est, 0.0, 0.0]
                fit  = minimize(self.chi_sq, fit_init, args=(), method='Nelder-Mead', jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None) 
                final_fit = [fit['x'][0], fit['x'][1], fit['x'][2], fit['x'][3]]  
                 
        else:
            if self.opt.LDClaw=='claret4':
                fit_init = [p_est, oot_est]
                fit  = minimize(self.chi_sq_no_gamma_4, fit_init, args=(self.binnedGamma[0][i], self.binnedGamma[1][i], self.binnedGamma[2][i], self.binnedGamma[3][i]), method='Nelder-Mead', jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None) 
                final_fit = [fit['x'][0], fit['x'][1], self.binnedGamma[0][i], self.binnedGamma[1][i],self.binnedGamma[2][i], self.binnedGamma[3][i]]      
            
            else:                       
                fit_init = [p_est, oot_est]
                fit  = minimize(self.chi_sq_no_gamma, fit_init, args=(self.binnedGamma[0][i], self.binnedGamma[1][i]), method='Nelder-Mead', jac=None, hess=None, hessp=None, bounds=None, constraints=(), tol=None, callback=None, options=None) 
                final_fit = [fit['x'][0], fit['x'][1], self.binnedGamma[0][i], self.binnedGamma[1][i]]      
                            
  
        p_err = np.sqrt(2)* (err/oot_est)/(np.sqrt(self.opt.n_exp/2.)) #optional err estimate on p
        final_fit.append(p_err)

        if self.opt.diagnostics == 1:
            if i == int(self.LC.shape[1]/2):
        
                if self.opt.timeline.apply_lo_dens_LC.val == 1:
                    if self.opt.LDClaw=='claret4':
                        model_curvefit   =   self.getModelLC_instant(final_fit[0], [final_fit[2], final_fit[3], final_fit[4], final_fit[5]])*final_fit[1]      
                    else: 
                        model_curvefit   =   self.getModelLC_instant(final_fit[0], [final_fit[2], final_fit[3]])*final_fit[1]          
                else:
                    if self.opt.LDClaw=='claret4':
                        model_curvefit   =   self.getModelLC_integrated(final_fit[0], [final_fit[2], final_fit[3], final_fit[4], final_fit[5]])*final_fit[1]  
                    else:
                        model_curvefit   =   self.getModelLC_integrated(final_fit[0], [final_fit[2], final_fit[3]])*final_fit[1]
                   
                
                time = self.opt.ndr_end_time[self.opt.effective_multiaccum-1::self.opt.effective_multiaccum]
    
                plt.figure('LC fit light curve %s microns'%(np.round(self.binnedWav[i],2)))    
                plt.ylabel('count in spectral bin (electrons)')
                plt.xlabel('time (sec)')
                plt.plot(time, ex, 'r+-')
                plt.plot(time, model_curvefit, 'bx-')
                plt.figure('LC fit normalised light curve %s microns'%(np.round(self.binnedWav[i],2)))
                plt.ylabel('count in spectral bin (electrons)')
                plt.xlabel('time (sec)')
                plt.ylim(1-0.015,1.001)
                plt.plot(time, ex/oot_est, 'r+-')
                plt.plot(time, model_curvefit/final_fit[1], 'bx-')
     
        return final_fit
        



#==============================================================================
# Extraction of spectrum and binning
#==============================================================================

class extractSpec():
    
    def __init__(self, data, opt, diff, ApFactor, final_ap):
        self.data = data
        self.opt = opt
        self.diff = diff # note this must not be substituted by opt.diff, since will vary for signal only, noise etc
        self.ApFactor = ApFactor
        self.final_ap = final_ap
             


    #==============================================================================
    # Apply a mask based on wavelength and F and extract 1-D spectrum
    #==============================================================================
    def applyMask_extract_1D(self):
        
        wl = self.opt.cr_wl.value
        F = self.opt.channel.camera.wfno_x.val
        pixSize = (self.opt.channel.detector_pixel.pixel_size.val).to(u.um).value
        ApFactor = self.ApFactor
        ApShape = self.opt.channel.data_pipeline.ApShape.val
        wl_max = self.opt.channel.data_pipeline.wavrange_hi.val
        
        jexosim_msg ("ap factor %s"%(ApFactor) , self.opt.diagnostics)
        jexosim_msg ("ap shape %s"%(ApShape ),      self.opt.diagnostics)
        jexosim_msg ("max wl %s"%(wl_max), self.opt.diagnostics)
        
 
           
#====================plot mask centre and edges========================================================== 

        w_list =[]
        for  i in range(len(wl)):
            if ApShape =='wav':                  
                    w = ApFactor*F*wl[i] 
            elif ApShape =='rect':  
                    w = ApFactor*F*wl_max  #defaults to rectangular
            w_list.append(w) # in distance units

        if self.diff !=1 :           
    #        self.opt.aa = self.data_signal_only.sum(axis =2)
            self.opt.aa = self.data.sum(axis =2)
             # 1) find max position of image
            indices = np.where(self.opt.aa == self.opt.aa.max())  
            y_max = indices[0].max() # max excludes rare situation of having 2 or more indices
            x_max = indices[1].max()
            ydata = self.opt.aa[:,x_max]
            xdata = np.arange(0,self.opt.aa.shape[0])            
            
            fitfunc  = lambda p, x: p[0]*np.exp(-0.5*((x-p[1])/p[2])**2)
            errfunc  = lambda p, x, y: (y - fitfunc(p, x))
            init  = [self.opt.aa[y_max,x_max], y_max , 2.0]
            out   = optimize.leastsq(errfunc, init, args=(xdata, ydata))[0]
    #                ymodel = out[0]*np.exp(-0.5*((xdata-out[1])/out[2])**2) 
            Cen0 = out[1]   # in pixel units     pixel unit = (distance unit/pixsize)  -0.5      
        else:
            Cen0 = (self.data.shape[0]/2.)-0.5
        
        jexosim_msg ("Cen0 %s"%(Cen0),  self.opt.diagnostics)
                           
        X1 = Cen0 - np.array(w_list)/pixSize
        X2 = Cen0 +  np.array(w_list)/pixSize
        
        X0 = [Cen0]* len(w_list)
        
        if self.final_ap == 1:
            self.opt.cen = Cen0
        

        if self.final_ap ==2 or self.final_ap ==1:   #excludes signal only and n_pix runs 
            jexosim_plot('y position of mask centre (pixels) vs x pixel', self.opt.diagnostics,
                         image = True,
                         image_data = self.data.sum(axis=2), aspect='auto', interpolation = None)    
        if self.final_ap ==2:   #2 = test ap chosen in noisy data 
            jexosim_plot('y position of mask centre (pixels) vs x pixel', self.opt.diagnostics,
                         ydata = X1, marker = 'b--')
            jexosim_plot('y position of mask centre (pixels) vs x pixel', self.opt.diagnostics,
                         ydata = X2, marker = 'b--')   
        if self.final_ap ==1:   #1 = final ap chosen in noisy data   
            jexosim_plot('y position of mask centre (pixels) vs x pixel', self.opt.diagnostics,
                         ydata = X0, marker = 'c--',  linewidth =3)  
            jexosim_plot('y position of mask centre (pixels) vs x pixel', self.opt.diagnostics,
                         ydata = X1, marker = 'w--',  linewidth =3)              
            jexosim_plot('y position of mask centre (pixels) vs x pixel', self.opt.diagnostics,
                         ydata = X2, marker = 'w--',  linewidth =3)              


#==============================================================================

        nWLs = self.data.shape[2]  # how many steps in the loop
        # Progress Bar setup:
        ProgMax = 100    # number of dots in progress bar
        if nWLs<ProgMax:   ProgMax = nWLs   # if less than 20 points in scan, shorten bar
        
        if self.final_ap == 1:
            print ("|" +    ProgMax*"-"    + "|     Applying variable position mask: progress")
            sys.stdout.write('|'); sys.stdout.flush();  # jexosim_msg start of progress bar
        nProg = 0   # fraction of progress   
               
    
        Cen_list=[]
        w_list =[]
        
        for i in range(self.data.shape[2]):
            if self.final_ap == 1:
                if ( i >= nProg*nWLs/ProgMax ):
     
                        sys.stdout.write('*'); sys.stdout.flush();  
                        nProg = nProg+1
                if ( i >= nWLs-1 ):
     
                        sys.stdout.write('|     done  \n'); sys.stdout.flush();                  
            inImage = self.data[...,i]
             # 1) find max position of image
            indices = np.where(inImage == inImage.max())  
            y_max = indices[0].max() # max excludes rare situation of having 2 or more indices
            x_max = indices[1].max()
            ydata = inImage[:,x_max]
            xdata = np.arange(0,inImage.shape[0])

#            if self.diff ==0:
#                fitfunc  = lambda p, x: p[0]*np.exp(-0.5*((x-p[1])/p[2])**2)
#                errfunc  = lambda p, x, y: (y - fitfunc(p, x))
#                init  = [inImage[y_max,x_max], y_max , 2.0]
#                out   = optimize.leastsq(errfunc, init, args=(xdata, ydata))[0]
##                ymodel = out[0]*np.exp(-0.5*((xdata-out[1])/out[2])**2) 
#                Cen = out[1]   #in pixel coords              
#            elif self.diff ==1:
#                Cen=Cen0*1.0
##                Cen = self.opt.aa.shape[0] /2. -0.5  
            
            Cen = Cen0*1.0
                            
            Cen_list.append(Cen)
            
            signal_list = [] 
            for j in range(inImage.shape[1]):                
                sig = inImage[: ,j]
                if ApShape =='wav':                  
                    w = ApFactor*F*wl[j] 
                elif ApShape =='rect':     
                    w = ApFactor*F*wl_max  #defaults to rectangular
                
                if i ==0:
                    w_list.append(2*w)
   
                X1 = Cen - w/pixSize
                X2 = Cen +  w/pixSize
                
                pixA = int(X1+0.5)
                pixB = int(X2+0.5)
                
                
                if pixA!= pixB:
                    wholepix = np.sum(sig[pixA+1:pixB])
                    RS = sig[pixB]*(X2+0.5-pixB)
                    LS = sig[pixA]*(pixA+0.5-X1)
                    in_mask_signal = wholepix +RS+LS
                    signal_list.append(in_mask_signal)
                                                   

                else:
                    in_mask_signal= sig[pixA]*(X2-X1)
                    signal_list.append(in_mask_signal)
                    
                
            if i == 0:
                signal_stack = signal_list
            else:
                signal_stack = np.vstack((signal_stack, signal_list))
 


        jexosim_plot('Centre of mask (pixel units) vs exposure', self.opt.diagnostics, 
             ydata=Cen_list)    
        
        jexosim_plot('width of mask (microns) vs pixel column', self.opt.diagnostics, 
             ydata=w_list, marker='bo')  
        
        jexosim_plot('width of mask (microns) vs wavelength of pixel column', self.opt.diagnostics, 
             xdata=wl, ydata = w_list, marker = 'bo' )     
        
  
 
        self.spectra=signal_stack



#==============================================================================
#  NIRISS curved mask
#==============================================================================

    def applyMask_extract_1D_NIRISS(self):
        
        # same mid position in all images
        
        wl = self.opt.cr_wl.value
        F = self.opt.channel.camera.wfno_x.val  # use F_x to set width, since NIRISS F_y is just estimate used to gen PSF if no WebbPSF used
        pixSize = (self.opt.channel.detector_pixel.pixel_size.val).to(u.um).value
        ApFactor = self.ApFactor
        ApShape = self.opt.channel.data_pipeline.ApShape.val
        wl_max = self.opt.channel.data_pipeline.wavrange_hi.val
        
        if self.opt.y_pos_osr != []:
            y_pos = self.opt.y_pos_osr[1::3]/3.0
           
             
        self.opt.aa = self.data.sum(axis =2)
        
        if self.final_ap ==1:   #final ap on noisy data 
            jexosim_plot('test - collapsed image stack', self.opt.diagnostics, 
                         image=True,
                         image_data = self.opt.aa)  
         
        Cen = self.opt.aa.shape[0] /2.

        y_pos = 23+ self.opt.y_pos_osr[1::3]/3.0 # factor 23 is empiric to match the centre of spectrum
        # Fix for NIRISS substrip96 again found empirically
        if self.data[...,0].shape[0] == 96 :
             y_pos = -97 + self.opt.y_pos_osr[1::3]/3.0
        
        
        mid_dist = pixSize/2.  + y_pos*pixSize 

        if self.final_ap ==2 or self.final_ap ==1:   #excludes signal only and n_pix runs          
            
            jexosim_plot('y position of mask centre (pixels) vs x pixel', self.opt.diagnostics,
                         image = True,
                         image_data = self.data.sum(axis=2), aspect='auto', interpolation = None)
            jexosim_plot('y position of mask centre (pixels) vs x pixel', self.opt.diagnostics,
                         ydata = y_pos, marker='yo')
         

   
        mid_dist00 = np.array([pixSize/2.  + Cen*pixSize]*len(wl))
        

        
        w_list =[]
        for  i in range(len(wl)):
            if ApShape =='wav':                  
                    w = ApFactor*F*wl[i] 
            elif ApShape =='rect':  
                    w = ApFactor*F*wl_max  #defaults to rectangular
            w_list.append(w)
        
        X1 = (mid_dist - np.array(w_list))/pixSize
        X2 = (mid_dist +  np.array(w_list))/pixSize
        
        for ii in range(len(X1)):
            X1[ii] = int(X1[ii])
  
  
        if self.final_ap ==2:   #2 = test ap chosen in noisy data 
            jexosim_plot('y position of mask centre (pixels) vs x pixel', self.opt.diagnostics,
                         ydata = X1, marker = 'b--')
            jexosim_plot('y position of mask centre (pixels) vs x pixel', self.opt.diagnostics,
                         ydata = X2, marker = 'b--')   
        if self.final_ap ==1:   #1 = final ap chosen in noisy data   
            jexosim_plot('y position of mask centre (pixels) vs x pixel', self.opt.diagnostics,
                         ydata = X1, marker = 'w--',  linewidth =3)              
            jexosim_plot('y position of mask centre (pixels) vs x pixel', self.opt.diagnostics,
                         ydata = X2, marker = 'w--',  linewidth =3)              





      
        nWLs = self.data.shape[2]  # how many steps in the loop
        # Progress Bar setup:
        ProgMax = 100    # number of dots in progress bar
        if nWLs<ProgMax:   ProgMax = nWLs   # if less than 20 points in scan, shorten bar
        print ("|" +    ProgMax*"-"    + "|     Applying variable position mask: progress")
        sys.stdout.write('|'); sys.stdout.flush();  # jexosim_msg start of progress bar
        nProg = 0   # fraction of progress   
                

        w_list =[]
        
        for i in range(self.data.shape[2]):
            if ( i >= nProg*nWLs/ProgMax ):
 
                    sys.stdout.write('*'); sys.stdout.flush();  
                    nProg = nProg+1
            if ( i >= nWLs-1 ):
 
                    sys.stdout.write('|     done  \n'); sys.stdout.flush();                  
            inImage = self.data[...,i]       
            
            signal_list = [] 
            for j in range(inImage.shape[1]):                
                sig = inImage[: ,j]
                if ApShape =='wav':                  
                    w = ApFactor*F*wl[j] 
                elif ApShape =='rect':  
                    w = ApFactor*F*wl_max  #defaults to rectangular
                
                if i ==0:
                    w_list.append(2*w)
   
                X1 = mid_dist[j] - w
                X2 = mid_dist[j] + w  
                
                
                pixA = int(X1/pixSize)
                pixB = int(X2/pixSize)
                
                
                if pixA!= pixB:
                    # find signal in whole pixels                 
                    in_mask_whole_pix = sig[pixA+1:pixB].sum()   
                    
                    # find fractional signal in X1 that lies in mask
                    x = (X1/pixSize - int(X1/pixSize)) * pixSize                
                    in_mask1 = ((pixSize-x)/pixSize)*sig[pixA]  
                    
                    # find fractional signal in X2 that lies in mask
                    x = (X2/pixSize - int(X2/pixSize)) * pixSize              
                    in_mask2 = (x/pixSize)*sig[pixB]
                  
                    in_mask_signal = in_mask_whole_pix  + in_mask1 + in_mask2               
                    signal_list.append(in_mask_signal)
                else:
                    in_mask_signal= sig[pixA]*(X2-X1)/pixSize
                    signal_list.append(in_mask_signal)
                    
            
                
            if i == 0:
                signal_stack = signal_list
            else:
                signal_stack = np.vstack((signal_stack, signal_list))
               
        jexosim_plot('width of mask (microns) vs pixel column', self.opt.diagnostics,
                     ydata = w_list, marker = 'bo')
        jexosim_plot('width of mask (microns) vs wavelength of pixel column', self.opt.diagnostics,
                     xdata=wl, ydata=w_list, marker='bo')

        self.spectra=signal_stack

     
        
    #==============================================================================
    # Extract 1-D spectrum
    #==============================================================================
        
    def extract1DSpectra(self):   
        
        spectra = np.zeros((self.data.shape[2],self.data.shape[1]))  
        # extract spectrum
        for i in range(self.data.shape[2]):
            spectra[i] = self.data[...,i].sum(axis = 0)
        
        self.spectra=spectra


    #==============================================================================
    # Binning of 1-D spectra into R- or fixed bins  
    #==============================================================================
        
    def binSpectra(self):
        
        wl = self.opt.cr_wl.value
        R = self.opt.channel.data_pipeline.R.val
        wavrange_lo = self.opt.channel.data_pipeline.wavrange_lo.val
        wavrange_hi = self.opt.channel.data_pipeline.wavrange_hi.val
        wavrange = [wavrange_lo, wavrange_hi]
        x_wav_osr = self.opt.x_wav_osr
        x_pix_osr = self.opt.x_pix_osr
        pixSize = (self.opt.channel.detector_pixel.pixel_size.val).to(u.um).value
        bin_size = self.opt.channel.data_pipeline.bin_size.val

        cond=1 #directs to new method
#       cond=0 # previous method
            
        
#        1) find the bin sizes in wavelength space
        if self.opt.channel.data_pipeline.binning.val == 'R-bin':
            jexosim_msg ('binning spectra into R-bins...',  self.opt.diagnostics)
            for i in range (len(wl)):
                if wl[i]>0:
                    idx0 = i
                    break
            for i in range (len(wl)-1,0,-1):
                if wl[i]>0:
                    idx1 = i
                    break
            wl0 = wl[idx0:idx1]
            if wl0[-1] < wl0[0]:
                w0 = wl0[-1] 
            else:
                w0 = wl0[0]
            dw = w0/(R-1)       
            bin_sizes=[dw]
            
            for i in range(500):
                dw2 = (1+1/(R-1))*dw
                bin_sizes.append(dw2)
                dw = dw2
                if np.sum(bin_sizes) > wavrange[1]-w0:
                    break
            bin_sizes = np.array(bin_sizes)   
            
#            2) find the edges of each bin in wavelength space
            wavcen = w0+np.cumsum(bin_sizes)
 
            wavedge1 = wavcen-bin_sizes/2.
            wavedge2 =  wavcen+bin_sizes/2.
            wavedge = np.hstack((wavedge1[0],((wavedge1[1:]+wavedge2[:-1])/2.)))
            
#            3)  find the edges in spatial space, in microns where 0 is at the left edge and centre of pixel is pixsize/2
#            # a) relate wl to x (microns)            
            wl_osr = x_wav_osr
            x_osr =  np.arange(pixSize/3./2., (pixSize/3.)*(len(x_wav_osr)), pixSize/3.)
            # this is the same as x_pix_osr but need the above if using cropped fp   
             # b) now interpolate this to find the spatial positions of edges of the bins
            xedge = interpolate.interp1d(wl_osr,x_osr, kind ='linear', bounds_error = False)(wavedge)            
            # positions of edges of bins in microns
            
#            4) remove nans

            x = np.arange(pixSize, pixSize*(len(wl)+1), pixSize)  # position of EDGES of pixels in microns starting at 0 on left side
            idx  =  np.argwhere(np.isnan(xedge))
            xedge0 = np.delete(xedge, idx)           
            wavcen0 = np.delete(wavcen, idx)
            
#            So now we have A) edges of bins in wavelength, B) edges of bins in x microns:xedge0
#            C) edges of pixels in x microns:x 
            
#            5) invert depending on wavelength solution 
            if wl0[-1]<wl0[0]:          
                xedge0 = xedge0[::-1]  
                wavcen0 = wavcen0[::-1]
            else:
                xedge0 = xedge0[1:]            
 

#==============================================================================
# old code
#==============================================================================
            if cond == 0: 
                b_pix =[]
                wavcen_list=[]
                ct = 0
                

                for i in range(len(x)):
                     cond2 =0
                    
 
                     if ct == len(xedge0)-1:
                         break
                     if xedge0[ct+1]- xedge0[ct] < pixSize:
                         jexosim_msg ("breaking since.. bin size < 1 pixel", self.opt.diagnostics)              
                         cond2 = 1               
                         break
                     
                     dist = x[i]
  
                     if dist > xedge0[ct]:
                         b_pix.append(i)
                         wavcen_list.append(wavcen0[ct])
                         ct = ct+1
                if cond2 ==1:        
                    jexosim_msg ("min wav before subpixel binning starts: %s"%(np.min(wavcen_list)),   self.opt.diagnostics)            
                    
                for exp in range(self.spectra.shape[0]): 
                                              
                    spec  = self.spectra[exp]
                                    
                    count =[] 
                    count0 = 0
                    ct = 0
                    for i in range (len(wl)):
                        if ct >= len(b_pix):
                            break
                        if i == b_pix[ct]:
                            ct=ct+1
                            count.append(count0)
                            count0 =0
                        else:
                            count0 = count0 + spec[i]
                    x = np.arange(0, pixSize*(len(wl)), pixSize) + pixSize/2.
        
                    for i in range(len(b_pix)-1):               
                        x1 = b_pix[i]*pixSize
                        x3 = (b_pix[i]+1)*pixSize
                        x2 = xedge0[i]                    
                        f= interpolate.interp1d(x, spec , kind ='linear', bounds_error = False)(np.array([x1,x2,x3]))
                        if f[1]+f[0] ==0:
                             f[1] =  f[0] = 1e-30
                        if f[2]+f[1] ==0:
                             f[2] =  f[1] = 1e-30                                                           
    #                    print f
                        A1 = abs((x2-x1)*(f[1]+f[0])/2. )
                        A2 = abs((x3-x2)*(f[2]+f[1])/2. )
                                  
                        S = spec[b_pix[i]]                 
                        S1 = S*A1/(A1+A2)
                        S2 = S*A2/(A1+A2)
    #                    print S, S1, S2                   
                        count[i] = count[i] +S1
                        count[i+1] = count[i+1] +S2   

                    wavcen_list0 = wavcen_list   
                    
                    # now bin for subpixel bins if they exist
                    if cond2 ==1:
                        ct=0                 
                        idx = np.argwhere(wavcen0<np.min(wavcen_list))                   
                        xedge1 = xedge0[idx].T[0]
                        wavcen1 = wavcen0[idx].T[0]
                        xedgepix = xedge1/pixSize    
                                                                           
                        # first add the right side of the first divided pixel to the final count bin     
                        fracRight = xedgepix[ct]-int(xedgepix[ct])
                        SRight = spec[int(xedgepix[ct])]*fracRight
                        count[-1] = count[-1]+SRight 
                        
                        for j in range(len(wavcen1)-1):
                  
                        
                            if int(xedgepix[ct+1]) > int(xedgepix[ct]):
                                
                                if int(xedgepix[ct+1]) == int(xedgepix[ct]) +1:
    #                                print "span"
                                    fracLeft = 1-(xedgepix[ct]-int(xedgepix[ct]))
                                    SLeft =  spec[int(xedgepix[ct])]*fracLeft
                                    fracRight = xedgepix[ct+1]-int(xedgepix[ct+1])
                                    SRight = spec[int(xedgepix[ct +1])]*fracRight
                                    S= SLeft+ SRight
                                    count.append(S)                             
                                    ct= ct+1                                                                    
                                else:
                                    qq= int(xedgepix[ct])
                                    temp=0
                                    fracLeft = 1- (xedgepix[ct]-int(xedgepix[ct]))
    #                                xxxx
                                    SLeft =  spec[qq]*fracLeft   
                                    temp +=SLeft
                                    for i in range(1000):
                                        qq = qq+1
                                        S = spec[qq]
    #                                    print "qq", qq
                                        temp +=S
                                        if xedgepix[ct+1] < qq+2:
                                            fracRight = (xedgepix[ct+1]-int(xedgepix[ct+1]))
                                            SRight = spec[qq+1]*fracRight
    #                                        print SRight
                                            temp +=SRight
                                            count.append(temp)
                                            ct=ct+1
                                            break
                                                                                                          
                            else:
                                frac = xedgepix[ct+1]-xedgepix[ct]
                                S = frac*spec[int(xedgepix[ct])]
                                count.append(S)
                                ct=ct+1      
                                              
                    
                    
                    wavcen_list0 = wavcen0[:-1] 
                    
                    jexosim_plot('binned spectrum', self.opt.diagnostics,
                                xdata = wavcen_list0, ydata = count, marker = 'bo') 
                    
           
                    if exp ==0:
                        count_array = count
                    else:
                        count_array = np.vstack((count_array, count))
#================ new code ==============================================================
# This deals with bins which are < 1 pixel

#=============   

            elif cond == 1:   
                for exp in range(self.spectra.shape[0]): 
                                          
                    spec  = self.spectra[exp]  # pick a 1 D spectrum
    
                    ct=0
                    count=[]
                    xedge1 = xedge0
                    wavcen1 = wavcen0 
                    xedgepix = xedge1/pixSize 
                    
                    for j in range(len(wavcen1)-1):
              
                        #selects if next bin edge is NOT in the same pixel
                        if int(xedgepix[ct+1]) > int(xedgepix[ct]):
                            
                            #selects if next bin edge is in the NEXT pixel
                            if int(xedgepix[ct+1]) == int(xedgepix[ct]) +1:
                                #signal from the left pixel
                                fracLeft = 1-(xedgepix[ct]-int(xedgepix[ct]))
                                SLeft =  spec[int(xedgepix[ct])]*fracLeft
                                #signal from the right pixel
                                fracRight = xedgepix[ct+1]-int(xedgepix[ct+1])
                                SRight = spec[int(xedgepix[ct +1])]*fracRight
                                # add these together
                                S= SLeft+ SRight
                                count.append(S)                             
                                ct= ct+1
                            
                            #selects if next bin edge is NOT in the NEXT pixel                                    
                            else:
                                qq= int(xedgepix[ct])
                                temp=0
                                #signal from the left pixel
                                fracLeft = 1- (xedgepix[ct]-int(xedgepix[ct]))
                                SLeft =  spec[qq]*fracLeft 
                                # add this to a cumulative count
                                temp +=SLeft
                                # move to the next pixel
                                for i in range(1000):
                                    qq = qq+1
                                    S = spec[qq]
                                    # add whole pixel count to cumulative
                                    temp +=S
                                    # check if next pixel has the bin edge
                                    if xedgepix[ct+1] < qq+2:
                                         # add the right pixel fraction to the count               
                                        fracRight = (xedgepix[ct+1]-int(xedgepix[ct+1]))
                                        SRight = spec[qq+1]*fracRight
                                        # final count for bin
                                        temp +=SRight
                                        
                                        count.append(temp)
                                        ct=ct+1
                                        break
                                                                                                      
                        else:
                            #selects if next bin edge is in SAME pixel
                            # find fraction of pixel in the bin 
                            frac = xedgepix[ct+1]-xedgepix[ct]
                            # add count
                            S = frac*spec[int(xedgepix[ct])]                           
                            count.append(S)
                            ct=ct+1
                    
                    wavcen_list0 = wavcen0[1:]                 
                    
                    jexosim_plot('binned spectrum', self.opt.diagnostics,
                                 xdata = wavcen_list0, ydata=count, marker = 'bo-')
                   
                        
           
                    if exp ==0:
                        count_array = count
                    else:
                        count_array = np.vstack((count_array, count))
   
            self.binnedLC = count_array
            self.binnedWav = wavcen_list0
            
         
        elif self.opt.channel.data_pipeline.binning.val  == 'fixed-bin':
            jexosim_msg ('binning spectra into fixed-bins of size %s pixel columns...%s'%(bin_size), self.opt.diagnostics)
            
            offs=0          
#============================use only temp for noise budget to match spectral ==================================================
            if self.opt.channel.name == 'NIRSpec_G395M':
                 offs = 5
            if self.opt.channel.name == 'MIRI_LRS':
                 offs = 3
                   
#==============================================================================
            spec = np.add.reduceat(self.spectra, np.arange(offs,self.spectra.shape[1])[::bin_size], axis = 1)
            wav = np.add.reduceat(wl, np.arange(offs,len(wl))[::bin_size]) / bin_size            
#            spec = np.add.reduceat(self.spectra, np.arange(self.spectra.shape[1])[::bin_size], axis = 1)
#            wav = np.add.reduceat(wl, np.arange(len(wl))[::bin_size]) / bin_size
            if wav[-1] < wav [-2]:
                wav = wav[0:-2]
                spec = spec[:,0:-2]
                            
            self.binnedLC = spec
            self.binnedWav = wav

    #==============================================================================
    # Binning of the LDCs   
    #==============================================================================


    def binGamma(self):
        
        wl = self.opt.cr_wl.value
        R = self.opt.channel.data_pipeline.R.val
        wavrange_lo = self.opt.channel.data_pipeline.wavrange_lo.val
        wavrange_hi = self.opt.channel.data_pipeline.wavrange_hi.val
        wavrange = [wavrange_lo, wavrange_hi]
        x_wav_osr = self.opt.x_wav_osr
        x_pix_osr = self.opt.x_pix_osr
        pixSize = (self.opt.channel.detector_pixel.pixel_size.val).to(u.um).value
        bin_size = self.opt.channel.data_pipeline.bin_size.val
        useWeightedAv =1 #better than using simple average
#        useWeightedAv =0 
        
        if self.opt.channel.data_pipeline.binning.val == 'R-bin':
            jexosim_msg ('binning LDCs into R-bins...', self.opt.diagnostics)
            for i in range (len(wl)):
                if wl[i]>0:
                    idx0 = i
                    break
            for i in range (len(wl)-1,0,-1):
                if wl[i]>0:
                    idx1 = i
                    break
            wl0 = wl[idx0:idx1]
            if wl0[-1] < wl0[0]:
                w0 = wl0[-1] 
            else:
                w0 = wl0[0]
            dw = w0/(R-1)       
            bin_sizes=[dw]
            for i in range(500):
                dw2 = (1+1/(R-1))*dw
                bin_sizes.append(dw2)
                dw = dw2
                if np.sum(bin_sizes) > wavrange[1]-w0:
                    break
            bin_sizes = np.array(bin_sizes)    
            wavcen = w0+np.cumsum(bin_sizes)
 
                       
            #positions in wavelength of bin edges        
            wavedge1 = wavcen-bin_sizes/2.
            wavedge2 =  wavcen+bin_sizes/2.
            wavedge = np.hstack((wavedge1[0],((wavedge1[1:]+wavedge2[:-1])/2.)))
            #positions in spatial microns of bin edges
            wl_osr = x_wav_osr
            x_osr =  np.arange(pixSize/3./2., (pixSize/3.)*(len(x_wav_osr)), pixSize/3.)
            xedge = interpolate.interp1d(wl_osr,x_osr, kind ='linear', bounds_error = False)(wavedge)            
            x = np.arange(pixSize, pixSize*(len(wl)+1), pixSize)   
            idx  =  np.argwhere(np.isnan(xedge))
            xedge0 = np.delete(xedge, idx)           
            wavcen0 = np.delete(wavcen, idx)
            
            if wl0[-1]<wl0[0]:          
                xedge0 = xedge0[::-1]  
                wavcen0 = wavcen0[::-1]
            else:
                xedge0 = xedge0[1:]            

        
            self.gamma = self.opt.ldc[1:]
            for jj in range(self.gamma.shape[0]):  
                                                
                spec  = self.gamma[jj]
                spec_flux  = np.sum(self.opt.fp_signal[1::3,1::3].value, axis=0) #used as weights for each coeff per pixel col.
                

#================ new code ==============================================================
# This deals with bins which are < 1 pixel
#=============          
                ct=0
                count=[]
                xedge1 = xedge0
                wavcen1 = wavcen0 
                xedgepix = xedge1/pixSize
                
                for j in range(len(wavcen1)-1):
          
                    
                    if int(xedgepix[ct+1]) > int(xedgepix[ct]):
                        
                        if int(xedgepix[ct+1]) == int(xedgepix[ct]) +1:
                            
                            if useWeightedAv !=1: 
                                fracLeft = 1-(xedgepix[ct]-int(xedgepix[ct]))
                                SLeft =  spec[int(xedgepix[ct])]*fracLeft                          
                                fracRight = xedgepix[ct+1]-int(xedgepix[ct+1])
                                SRight = spec[int(xedgepix[ct +1])]*fracRight
                                S= SLeft+ SRight
                                binsize = xedgepix[ct+1] - xedgepix[ct]
                                S= S/binsize
                            elif useWeightedAv ==1:  #weights each coeff by the flux in that pixel column
                                # Weighted average
                                SLeft_flux = spec_flux[int(xedgepix[ct])]*fracLeft
                                SRight_flux = spec_flux[int(xedgepix[ct])]*fracRight
                                SLeft =  spec[int(xedgepix[ct])]
                                SRight = spec[int(xedgepix[ct +1])]
                                S = (SLeft*SLeft_flux + SRight*SRight_flux)/ (SLeft_flux + SRight_flux)    

                            # add S                 
                            count.append(S)                             
                            ct= ct+1                           
                            
                                                            
                        else:
                            if useWeightedAv !=1: 
                                #simple average
                                qq= int(xedgepix[ct])
                                temp=0
                                binsize_temp = 0
                                fracLeft = 1- (xedgepix[ct]-int(xedgepix[ct]))                                
                                SLeft =  spec[qq]*fracLeft   
                                temp +=SLeft
                                binsize_temp += fracLeft
                                for i in range(1000):
                                    qq = qq+1
                                    S = spec[qq]
    #                              
                                    temp +=S
                                    binsize_temp += 1
                                    if xedgepix[ct+1] < qq+2:
                                                       
                                        fracRight = (xedgepix[ct+1]-int(xedgepix[ct+1]))
                                        SRight = spec[qq+1]*fracRight
    #                                        
                                        temp +=SRight
                                        binsize_temp += fracRight                      
                                        count.append(temp/binsize_temp)
                                        ct=ct+1
                                        break
                            elif useWeightedAv ==1:                                                                                                   
                                #weighted average : make two lists S for gamma*flux(i.e. weight) and flux_list for flux
                                flux_list =[] #weights
                                S_list = []
                                qq= int(xedgepix[ct])
                                fracLeft = 1- (xedgepix[ct]-int(xedgepix[ct]))
                                SLeft =  spec[qq]                             
                                SLeft_flux =  spec_flux[qq]*fracLeft
                                S_list.append(SLeft*SLeft_flux) #S*weight
                                flux_list.append(SLeft_flux)
                                for i in range(1000):
                                    qq = qq+1
                                    S = spec[qq]
                                    S_flux = spec_flux[qq]
                                    S_list.append(S*S_flux) #S*weight
                                    flux_list.append(S_flux)
                                    
                                    if xedgepix[ct+1] < qq+2:                                                  
                                        fracRight = (xedgepix[ct+1]-int(xedgepix[ct+1]))
                                        SRight = spec[qq+1]
                                        SRight_flux = spec_flux[qq+1]*fracRight
                                        
                                        S_list.append(SRight*SRight_flux) #S*weight
                                        flux_list.append(SRight_flux)
                                        weighted_av = np.sum(S_list)/np.sum(flux_list)                     
                                        count.append(weighted_av)
                                        ct=ct+1
                                        break
                                                                                                  
                    else:
#                        frac = xedgepix[ct+1]-xedgepix[ct]
                        S = spec[int(xedgepix[ct])]  # if bin size <1 pixel and within the pixel, no weighting needed                        
                        count.append(S)
                        ct=ct+1                              
                        
                jexosim_plot('binned Gamma', self.opt.diagnostics,
                             xdata=self.opt.ldc[0], ydata=self.opt.ldc[jj+1], marker = 'ro-')
                jexosim_plot('binned Gamma', self.opt.diagnostics,
                             xdata=self.binnedWav, ydata=count, marker ='bo-')         
    
  
           
                if jj ==0:
                    count_array = count
                else:
                    count_array = np.vstack((count_array, count))
   
            self.binnedGamma = count_array
            
         
        elif self.opt.channel.data_pipeline.binning.val  == 'fixed-bin':
            jexosim_msg ('binning gamma into fixed-bins of size %s pixel columns...%s'%(bin_size), self.opt.diagnostics)
            self.gamma = self.opt.ldc[1:]
            for jj in range(self.gamma.shape[0]):            
                spec  = self.gamma[jj]           
                spec = np.add.reduceat(spec, np.arange(len(spec))[::bin_size]) / bin_size
                wav = np.add.reduceat(wl, np.arange(len(wl))[::bin_size]) / bin_size
                if wav[-1] < wav [-2]:
                    wav = wav[0:-2]
                    spec = spec[0:-2]                    
                if jj ==0:
                    count_array = spec
                else:
                    count_array = np.vstack((count_array, spec))
                            
            self.binnedGamma = count_array


    def binGamma2(self):
        
        wl = self.opt.cr_wl.value
        R = self.opt.channel.data_pipeline.R.val
        wavrange_lo = self.opt.channel.data_pipeline.wavrange_lo.val
        wavrange_hi = self.opt.channel.data_pipeline.wavrange_hi.val
        wavrange = [wavrange_lo, wavrange_hi]
        x_wav_osr = self.opt.x_wav_osr
        x_pix_osr = self.opt.x_pix_osr
        pixSize = (self.opt.channel.detector_pixel.pixel_size.val).to(u.um).value
        bin_size = self.opt.channel.data_pipeline.bin_size.val        
        
        
        if self.opt.channel.data_pipeline.binning.val == 'R-bin':
            for i in range (len(wl)):
                if wl[i]>0:
                    idx0 = i
                    break
            for i in range (len(wl)-1,0,-1):
                if wl[i]>0:
                    idx1 = i
                    break
            wl0 = wl[idx0:idx1]

            if wl0[-1] < wl0[0]:
                w0 = wl0[-1] 
            else:
                w0 = wl0[0]
            dw = w0/(R-1)
      
            bin_sizes=[dw]
            for i in range(500):
                dw2 = (1+1/(R-1))*dw
                bin_sizes.append(dw2)
                dw = dw2
                if np.sum(bin_sizes) > wavrange[1]-w0:
                    break
            bin_sizes = np.array(bin_sizes)    
            wavcen = w0+np.cumsum(bin_sizes)
                  
            #positions in wavelength of bin edges
            wavedge1 = wavcen-bin_sizes/2.
            wavedge2 =  wavcen+bin_sizes/2.
            wavedge = np.hstack((wavedge1[0],((wavedge1[1:]+wavedge2[:-1])/2.)))
         
            #positions in spatial microns of bin edges      
            wl_osr = x_wav_osr
            x_osr =  np.arange(pixSize/3./2., (pixSize/3.)*(len(x_wav_osr)), pixSize/3.)
            # this is the same as x_pix_osr but need the above if using cropped fp
 
 
                    
            xedge = interpolate.interp1d(wl_osr,x_osr, kind ='linear', bounds_error = False)(wavedge)         
            x = np.arange(pixSize, pixSize*(len(wl)+1), pixSize)
            idx  =  np.argwhere(np.isnan(xedge))
            xedge0 = np.delete(xedge, idx)
            wavcen0 = np.delete(wavcen, idx)
                      
            if wl0[-1]<wl0[0]:          
                xedge0 = xedge0[::-1]  
                wavcen0 = wavcen0[::-1]
            else:
                xedge0 = xedge0[1:]
            
            b_pix =[]
            wavcen_list=[]
            xedge00=[]
            ct = 0
            for i in range(len(x)):
#               
                if ct == len(xedge0)-1:
                    jexosim_msg ("break due to complete range covered..." , self.opt.diagnostics)
                    break
                if xedge0[ct+1]- xedge0[ct] < pixSize: 
                    jexosim_msg ("break due to small pixel size...",  self.opt.diagnostics)
                    break
                
                dist = x[i]
                if dist > xedge0[ct]:
                    b_pix.append(i)
                    wavcen_list.append(wavcen0[ct])
                    xedge00.append(xedge0[ct])
                    ct = ct+1                         
            xedge00.append(xedge0[ct])
         
            self.gamma = self.opt.ldc[1:]
            for jj in range(self.gamma.shape[0]):            
                spec  = self.gamma[jj]
                count =[] 
                count0 = 0
                ct = 0
                for i in range (len(wl)):
                    if ct >= len(b_pix):
                        break
                    if i == b_pix[ct]:
                        ct=ct+1
                        count.append(count0)
                        count0 =0
                    else:
                        count0 = count0 + spec[i]
                        
                x = np.arange(0, pixSize*(len(wl)), pixSize) + pixSize/2.
                for i in range(len(b_pix)-1):              
                    x1 = b_pix[i]*pixSize
                    x3 = (b_pix[i]+1)*pixSize
#                    x2 = xedge0[i]
                    x2 = xedge00[i]
                    f= interpolate.interp1d(x, spec , kind ='linear', bounds_error = False)(np.array([x1,x2,x3]))                
                    if f[1]+f[0] ==0:
                         f[1] =  f[0] = 1e-30
                    if f[2]+f[1] ==0:
                         f[2] =  f[1] = 1e-30                      
                    A1 = (x2-x1)*(f[1]+f[0])/2. 
                    A2 = (x3-x2)*(f[2]+f[1])/2.                     
                    S = spec[b_pix[i]]                  
                    S1 = S*A1/(A1+A2)
                    S2 = S*A2/(A1+A2)
                    count[i] = count[i] +S1
                    count[i+1] = count[i+1] +S2
                
                bin_sizes  = np.hstack((xedge00[0],np.diff(xedge00)))/pixSize
                count = count/bin_sizes[:-1]
                if jj ==0:
                    count_array = count
                else:
                    count_array = np.vstack((count_array, count))
                
                jexosim_plot()    
                plt.figure('binned Gamma', self.opt.diagnostics,
                           xdata = self.opt.ldc[0], ydata = self.opt.ldc[jj+1], marker = 'ro-')
                plt.figure('binned Gamma', self.opt.diagnostics,
                           xdata = wavcen_list, ydata = count, marker = 'bo-')
                              
                
            self.binnedGamma = count_array
 

        elif self.opt.channel.data_pipeline.binning.val  == 'fixed-bin':
            jexosim_msg ('binning gamma into fixed-bins of size %s pixel columns...%s'%(bin_size), self.opt.diagnostics)
            self.gamma = self.opt.ldc[1:]
            for jj in range(self.gamma.shape[0]):            
                spec  = self.gamma[jj]           
                spec = np.add.reduceat(spec, np.arange(len(spec))[::bin_size]) / bin_size
                wav = np.add.reduceat(wl, np.arange(len(wl))[::bin_size]) / bin_size
                if wav[-1] < wav [-2]:
                    wav = wav[0:-2]
                    spec = spec[0:-2]                    
                if jj ==0:
                    count_array = spec
                else:
                    count_array = np.vstack((count_array, spec))
                            
            self.binnedGamma = count_array
            
#==============================================================================
# processing of OOT simulations                    
#==============================================================================
class processOOT():
          
     def __init__(self, LC, LC_signal, binnedWav, opt):
        self.LC = LC
        self.LC_signal = LC_signal
        self.binnedWav = binnedWav
        self.opt = opt      
        self.obtainSNR() 
          
        if self.opt.channel.data_pipeline.useAllen.val ==1:
            for i in [0]:
                self.obtainAllen() 

                                
                time_target = self.opt.T14.value
#                time_target = 3600               
                idx = 0
                if self.ootAllen[0].max() > time_target:
                    idx = np.argwhere(self.ootAllen[0] > time_target)[0].item()
                    idx = idx-50 # how many indicies from the end for fit
                    if idx < 0:
                        idx = 0
                t1 = self.ootAllen[0][idx]
                               
                #  starting point
                idx = int(len(self.ootAllen[0])*0.25)
                t1 = self.ootAllen[0][idx]                                
                                                
                x_fit = np.log10(self.ootAllen[0][idx:])
                y_fit = np.log10(self.ootAllen[3][idx:])      
                init = [-2.2,-0.5]
                res_lsq = optimize.leastsq(self.linear2, init, args=(x_fit, y_fit))
                c =  res_lsq[0][0]
                m = res_lsq[0][1]
                fit_time = np.arange(t1,time_target,10)
                fit_y= 10**(c)*(fit_time)**(m)

                
                no= []
                for ii in range(self.ootAllen[2].shape[1]):                 
                    x_fit = np.log10(self.ootAllen[0][idx:])
                    y_fit = np.log10(self.ootAllen[2][:,ii][idx:])             
                    init = [-2.2,-0.5]
                    if y_fit[0]<1:
                        m_est = (y_fit[-1]-y_fit[0]) / (x_fit[-1]-x_fit[0])                        
                        init = [y_fit[0],-0.1]   
                    res_lsq = optimize.leastsq(self.linear2, init, args=(x_fit, y_fit))
                    c =  res_lsq[0][0]
                    m = res_lsq[0][1]
                    fit_time = np.arange(t1,time_target,10)
                    fit_y= 10**(c)*(time_target)**(m)*1e6
                    fit_y_= 10**(c)*(fit_time)**(m)
    
                    no.append(fit_y)
                    
                self.noiseAt1hr = np.array(no)
           
     def obtainSNR(self):
        self.ootSignal = self.LC_signal.mean(axis=0)
        self.ootNoise = self.LC.std(axis=0)
    
     def linear2(self, init, x,y):
        c = init[0]
        m = init[1]
        err = y**2 - (m*x+c)**2
        return err          
     def obtainAllen(self):
         
        timestep = self.opt.exposure_time.value
        binMax = int(self.LC.shape[0]/20  )
        for binSize in range (1, binMax):
            idx = np.arange(0, self.LC.shape[0],binSize)
            sig = (np.add.reduceat(self.LC,idx,axis =0)/ binSize)[:-1]
#            noise = sig.std(axis=0) / sig.mean(axis=0)
            noise = sig.std(axis=0) / self.LC_signal.mean(axis=0)
            if binSize ==1 :
#                sig_stack = sig.mean(axis=0)
                sig_stack = self.LC_signal.mean(axis=0)
                no_stack = noise
            else:
#                sig_stack = np.vstack((sig_stack,sig.mean(axis=0)))
                sig_stack = np.vstack((sig_stack,self.LC_signal.mean(axis=0)))            
                no_stack = np.vstack((no_stack,noise))
        no_median = np.median(no_stack, axis =1)              
        binTimes = np.arange(1, binMax)* timestep
        self.ootAllen = [binTimes, sig_stack, no_stack, no_median]
