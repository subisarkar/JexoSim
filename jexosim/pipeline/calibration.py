"""
JexoSim
2.0
JPD calibration module
v1.0

"""

import numpy as np
from jexosim.lib.jexosim_lib import jexosim_msg


def loadData(opt):
    return opt    
    
def dqInit(data, opt):  
    dq_array = np.zeros(data.shape)
    for ii in range(data.shape[2]):
        dq_array[...,ii] = dq_array[...,ii] + opt.init_pix
    # flag 1 = initially known 'inoperable pixels'
#    plt.figure('dq array')
#    plt.imshow(dq_array[...,0])        
    opt.dq_array = dq_array
    return opt.dq_array     
    

def satFlag(data, opt): 
    jexosim_msg ("flagging staurated pixels...", opt.diagnostics)
    sat_limit = opt.channel.detector_pixel.full_well.val
    idx = np.argwhere(data > sat_limit)
    dq_array = opt.dq_array
    for i in range (len(idx)):
        dq_array[idx[i][0]][idx[i][1]][idx[i][2]] = 2      
    # flag 2 = 'saturated pixels' (overmaps flag 1)    
#    plt.figure('dq array')
#    plt.imshow(dq_array[...,0])
    jexosim_msg ("number of saturated pixels over all NDRs %s"%(len(idx) ),  opt.diagnostics)
    opt.dq_array = dq_array           
    return opt.dq_array     
 

def badCorr(data, opt): 
    jexosim_msg ("correcting bad pixels...", opt.diagnostics)
    dq_array = opt.dq_array
#    plt.figure('data 00')
#    plt.imshow(data[...,0])   
    idx = np.argwhere(dq_array > 0)
 
    for i in range (len(idx)):
        data[idx[i][0]][idx[i][1]][idx[i][2]] = 0   
#    plt.figure('data 01')
#    plt.imshow(data[...,0], interpolation ='None')    
        
    return data      
    
      
def subZero(opt):
    jexosim_msg ("subtracting zeroth read...", opt.diagnostics)
    multiaccum = opt.effective_multiaccum
    
    new_data = np.zeros((opt.data.shape[0],opt.data.shape[1],opt.data.shape[2]))
      
    for i in range (0, opt.data.shape[2], multiaccum):
        for j in range (0,multiaccum):
            new_data[...,i+j] = opt.data[...,i+j] - opt.data[..., i]
    opt.data = new_data        
    return opt   
        
def subDark(opt):    
    jexosim_msg ("subtracting dark signal...", opt.diagnostics)
    
    dc_time =  opt.channel.detector_pixel.Idc.val* opt.duration_per_ndr
    new_data = opt.data - dc_time   
    opt.data = new_data
    return opt
        
    
def flatField(opt):    
    jexosim_msg ("applying flat field...", opt.diagnostics)
    
    QE_grid = opt.qe_grid   
    opt.data =  np.rollaxis(opt.data,2,0)
    opt.data = opt.data/QE_grid
    opt.data =  np.rollaxis(opt.data,0,3) 
    
    jexosim_msg("std of flat field...%s"%(QE_grid.std()), opt.diagnostics)
    
    return opt
    

def subBackground(opt) : 
    jexosim_msg ("subtracting background...", opt.diagnostics)
    
    if opt.channel.data_pipeline.use_fast.val ==1 and opt.channel.instrument.val!='NIRISS' and \
       opt.data.shape[0]>20:
           aa = opt.bkg.sum(axis=0)/opt.bkg.shape[0] #mean value per column
           opt.data = opt.data- aa                     
    else:
        border_pix = 5
        if opt.data.shape[0]<20:
               border_pix = 1
        background_1 = opt.data[0:border_pix]
        background_2 = opt.data[-border_pix:]   
        background = np.vstack ( (background_1, background_2) )     
        aa = background.sum(axis=0)/background.shape[0]
        opt.data = opt.data- aa

    return opt
    
def doUTR(data, opt):
       
    multiaccum = int(opt.effective_multiaccum)
    n_exp = opt.n_exp
    time =  opt.ndr_end_time.value
    
    if multiaccum==2 or opt.channel.detector_readout.doCDS.val ==1:
        jexosim_msg ("doing CDS...", opt.diagnostics)

        new_data = np.zeros((data.shape[0],data.shape[1], n_exp))
        ct = 0        
        for i in range (multiaccum-1, data.shape[2], multiaccum):
            new_data[...,ct] = data[...,i]-data[...,i-multiaccum+1]
            ct = ct+1
        return new_data
    else:
        jexosim_msg ("fitting ramps...", opt.diagnostics)
        utr_data = np.zeros(( data.shape[0], data.shape[1] ,n_exp))
#        utr_error = np.zeros(( data.shape[0], data.shape[1] ,n_exp))
   
        x = data.shape[1]/2
        y = data.shape[0]/2
        
        t_ndr = time[0:multiaccum]
    
        ct=0
        for i in range (0, n_exp*multiaccum, multiaccum):
              
            a = data[...,i:i+multiaccum]
            
#            Mx = t_ndr.mean()
#            My = a.mean(axis=2)
            Sx = t_ndr.std()
            Sy = a.std(axis=2)
            R = (len(t_ndr)*(t_ndr*a).sum(axis=2) - t_ndr.sum()*a.sum(axis=2))  / \
            (np.sqrt( (len(t_ndr)*np.sum(t_ndr**2) - (np.sum(t_ndr))**2) * ( len(t_ndr)*(a**2).sum(axis=2) - (a.sum(axis=2))**2) ))
             
            m = R*Sy/Sx
#            c = My - m*Mx
#            m_ = np.ones((m.shape[0], m.shape[1], len(t_ndr)))
#            c_ = np.ones((c.shape[0], c.shape[1], len(t_ndr))) 
#            
#            for j in range(len(t_ndr)):
#                m_[...,j] = m
#                c_[...,j] = c
#                
#            y_model = m_*t_ndr+c_
#            
#            residuals = ((y_model-a)**2).sum(axis=2)
#            
#            n = len(t_ndr)
#            D = sum(t_ndr**2) - 1./n * sum(t_ndr)**2
#            x_bar = np.mean(t_ndr)
#            dm_squared = 1./(n-2)*residuals/D
#            dc_squared = 1./(n-2)*(D/n + x_bar**2)*residuals/D
#            dm = np.sqrt(dm_squared)
#            dc = np.sqrt(dc_squared)
         
#             plot central pixel ramp and fit
#            print m[y][x], c[y][x], dm[y][x], dc[y][x],  R[y][x]
            
#            plt.figure('slope')
#            plt.plot(t_ndr, a[y][x], 'ro')
#            plt.plot(t_ndr, y_model[y][x], 'k--')
#            plt.grid(True)
#            plt.xlabel('time (s)')
#            plt.ylabel('electrons')
            
            utr_data[...,ct] = m
#            utr_error[...,ct] = dm
            ct +=1
        # if zero background used, then nan created with slopes - messes up later steps.
        utr_data[np.isnan(utr_data)] = 0  
#        utr_error[np.isnan(utr_error)] = 0 
        
        utr_data *= opt.t_int
         
#        return utr_data, utr_error   
          
        return utr_data