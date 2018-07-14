import warnings
import numpy as np
import math

import utils.imtools as imtools

def fft_lowpass(img_in, cpd_cutoff, stim_cpd, beta=None, alpha=0.05, rescale=True):
    '''
    Lowpass filter an image at a given cpd cuttoff using Fourier representation, for a given cpd of the img stimulus
    
    Args:
        img_in (2d numpy array):   stimluius img
        cpd_cuttoff (float):  maximum CPD value present in output img
        stim_cpd (float):    CPD of stimulus (should be larger than cpd_cuttoff)
        beta (int):        shape parameter for gneralized gaussian - if None then do hard cuttoff.
        alpha (float):      width parameters for genrallized gaussian (default 1)
        rescale (bool):
        
    Returns:
        stim (2d numpy array):   stimlulus image fourier filtered and no frequencies higher than cpd_cuttoff
        mag (2d numpy array):   magnitude of stimulus
        phase (2d numpy array): global phase angle of stimulus
        filt (2d numpy array): fiter used to create filtered img
    '''
    #make sure parameters make sense
    if cpd_cutoff > stim_cpd:
        warnings.warn('Cutoff CPD is higher than stimulus CPD')
    
    #find ratio of cuttoff to max cpd so we know where to stop in fourier space
    fft_diameter_fc = cpd_cutoff/(stim_cpd)
    fft = np.fft.fftshift(np.fft.fft2(img_in))
    mag = np.abs(fft)
    xx, yy = np.meshgrid(np.linspace(-1, 1, img_in.shape[0]),
                         np.linspace(-1, 1, img_in.shape[1])) #THIS IS NOT VALID IF STIM IS NOT SQUARE111
    fft_diameters = np.sqrt(xx**2 + yy**2)
    
    #cuttoff amplitude for frequencies above cuttoff based on gernalized gaussian
    if(beta==None):
    # Anything greater than the cpd_cutoff will be set to 0 in the mag
        filt = (fft_diameters <= fft_diameter_fc)
        mag = np.multiply(mag, filt)
    else:
        #calculate half width at half max: relationship between cuttoff fq and frequency where gauss taper is centerd.
        hwhm = alpha * (np.log(2))**(1./beta)
        #center of gaussian is cuttoff minus half witch half max
        fft_diameter_fd = fft_diameter_fc - hwhm
        
        if fft_diameter_fd < 0:
            warnings.warn('Taper Top Negative - Won\'t reach full contrast.')
        #generic gauusian scaing function
        filt = np.exp(-1*(np.abs(fft_diameters-fft_diameter_fd)/alpha)**beta)
        filt[fft_diameters<fft_diameter_fd] = 1.
        mag = np.multiply(mag,filt)
    
    phase = np.angle(fft)
    
    # Reconstruct the image using the inverse fourier transform
    ifft = np.fft.fftshift(mag * np.exp(phase * 1.0j))
    img_ifft = np.fft.ifft2(ifft).real
    
    # Rescale to [0,255]
    if(rescale):
        img_ifft = imtools.rescale_255(img_ifft)
    
    return img_ifft, mag, phase, filt