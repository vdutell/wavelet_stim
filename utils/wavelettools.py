import numpy as np
import warnings

def wavelet_lowpass(img_in, cpd_cuttoff, stim_cpd, wavelet='Gaussian'):
    '''
    Lowpass filter an image at a given cpd cuttoff, using wavelet representation for a given cpd of the img stimulus
    
    Args:
        img_in (2d numpy array):   stimluius img
        cpd_cuttoff (float):  maximum CPD value present in output img
        stim_cpd (float):    CPD of stimulus (should be larger than cpd_cuttoff)
        
    Returns:
        stim (2d numpy array):   stimlulus image fourier filtered and no frequencies higher than cpd_cuttoff
    '''
    
    
    warnings.warn('Wavelets Not Yet Implemented! Returning Unfiltered Image.')
    return(img_in)