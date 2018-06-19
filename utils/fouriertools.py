import warnings
import numpy as np

import utils.imtools as imtools

def fft_lowpass(img_in, cpd_cutoff, stim_cpd, rescale=True):
    '''
    Lowpass filter an image at a given cpd cuttoff using Fourier representation, for a given cpd of the img stimulus
    
    Args:
        img_in (2d numpy array):   stimluius img
        cpd_cuttoff (float):  maximum CPD value present in output img
        stim_cpd (float):    CPD of stimulus (should be larger than cpd_cuttoff)
        
    Returns:
        stim (2d numpy array):   stimlulus image fourier filtered and no frequencies higher than cpd_cuttoff
        mag (2d numpy array):   magnitude of stimulus
        phase (2d numpy array): global phase angle of stimulus
    '''
    #make sure parameters make sense
    if cpd_cutoff > stim_cpd:
        warnings.warn('Cutoff CPD is higher than stimulus CPD')
    
    #find ratio of cuttoff to max cpd so we know where to stop in fourier space
    fft_diameter = cpd_cutoff/(stim_cpd)
    fft = np.fft.fftshift(np.fft.fft2(img_in))
    mag = np.abs(fft)
    xx, yy = np.meshgrid(np.linspace(-1, 1, img_in.shape[0]),
                         np.linspace(-1, 1, img_in.shape[1]))
    dist = np.sqrt(xx**2 + yy**2)
    
    # Anything greater than the cpd_cutoff will be set to 0 in the mag
    mag = mag * (dist <= fft_diameter)
    phase = np.angle(fft)
    
    # Reconstruct the image using the inverse fourier transform
    ifft = np.fft.fftshift(mag * np.exp(phase * 1.0j))
    img_ifft = np.fft.ifft2(ifft).real
    
    # Rescale to [0,255]
    if(rescale):
        img_ifft = imtools.rescale_255(img_ifft)
    
    return img_ifft, mag, phase