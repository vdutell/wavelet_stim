import warnings
import numpy as np
import math

import utils.imtools as imtools



def fft_lowpass(img_in, cpd_cutoff, stim_cpd, filt='sharp', rescale=True):
    '''
    Lowpass filter an image at a given cpd cuttoff using Fourier representation, for a given cpd of the img stimulus
    
    Args:
        img_in (2d numpy array):   stimluius img
        cpd_cuttoff (float):  maximum CPD value present in output img
        stim_cpd (float):    CPD of stimulus (should be larger than cpd_cuttoff)
        filt (str):        define the type of filtering desired (sharp, cosine_step, gauss_step, gauss_taper)
        
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
    
    #calculate filter
    if(filt=='sharp'):
    # Anything greater than the cpd_cutoff will be set to 0 in the mag
        filt = (fft_diameters <= fft_diameter_fc)
        mag = np.multiply(mag, filt)
        
    elif(filt=='gauss_taper'):
        #calculate sigma needed for HWHM value to be at fc
        sigma = fft_diameter_fc / (np.sqrt(-np.log(np.sqrt(0.5))))
        filt = np.exp(-1*(fft_diameters/sigma)**2)
        mag = np.multiply(mag,filt)
        
    elif(filt=='cosine_step'):
        #cosine step taper, full power at fd, zero power at 2*fd
        
        #calc fd (taper start) and fz (taper end)
        fd = fft_diameter_fc*np.pi/(np.arccos(np.sqrt(2)-1)+np.pi)
        #end taper at 2*fd for a power scale
        fz = 2*fd
        #can now define function
        filt = 0.5*(1+np.cos(np.pi*(fft_diameters-fd)/(fz-fd)))

        filt[fft_diameters < fd] = 1
        filt[fft_diameters > fz] = 0
        
        #multiply by filter
        mag = np.multiply(mag,filt)
    
    elif(filt=='gauss_step'):
        #cuttoff amplitude for frequencies above cuttoff based on gernalized gaussian
        beta=2 #gaussian for now (beta=1 for Laplacian)
        alpha=0.05
        #calculate half width at half max: relationship between cuttoff fq and frequency where gauss taper is centerd.
        hwhm = alpha * (np.log(2))**(1./beta)
        #center of gaussian is cuttoff minus half witch half max
        fft_diameter_fd = fft_diameter_fc - hwhm
        #check if top of gaussian is negative.
        if fft_diameter_fd < 0:
            warnings.warn('Taper Top Negative - Won\'t reach full contrast.')
        #generic gauusian scaing function
        filt = np.exp(-1*(np.abs(fft_diameters-fft_diameter_fd)/alpha)**beta)
        filt[fft_diameters<fft_diameter_fd] = 1.
    else:
        raise ValueError(f'{filt} is an unknown filtering type! Currently Supported Decompositions are \'fourier_sharp\', \'fourier_gauss\' and \'wavelet\'.  Returning original Signal.')
        filt = 1
    #multioly by fourier filter
    mag = np.multiply(mag,filt)
    phase = np.angle(fft)
    
    # Reconstruct the image using the inverse fourier transform
    ifft = np.fft.fftshift(mag * np.exp(phase * 1.0j))
    img_ifft = np.fft.ifft2(ifft).real
    
    # Rescale to [0,255]
    if(rescale):
        img_ifft = imtools.rescale_255(img_ifft)
    
    return img_ifft, mag, phase, filt