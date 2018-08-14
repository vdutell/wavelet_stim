import warnings
import numpy as np
import math

import utils.imtools as imtools

import matplotlib.pyplot as plt

# from https://code.google.com/archive/p/agpy/downloads
def azimuthalAverage(image, center=None):
    """
    Calculate the azimuthally averaged radial profile.

    Parameters
    ----------
    image:   2d numpy array defining The 2D image
    center:  The [x,y] pixel coordinates used as the center. The default is 
             None, which then uses the center of the image (including 
             fracitonal pixels).
             
    Returns:
    --------
    radial_prof: 1D nupy array defining the circularly averaged value of 2d image
    
    """
    # Calculate the indices from the image
    #print(np.indices(image.shape))
    y, x = np.indices(image.shape)
    
    #shape of image
    framew = np.shape(image)[0] #in pixels
    frameh = np.shape(image)[1] #in pixels

    if not center:
        center = np.array([(x.max()-x.min())/2.0, (x.max()-x.min())/2.0])

    r = np.hypot(x - center[0], y - center[1])

    # Get sorted radii
    ind = np.argsort(r.flat)
    r_sorted = r.flat[ind]
    i_sorted = image.flat[ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.astype(int)

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented
    rind = np.where(deltar)[0]       # location of changed radius
    nr = rind[1:] - rind[:-1]        # number of radius bin
    
    # Cumulative sum to figure out sums for each radius bin
    csim = np.cumsum(i_sorted, dtype=complex)
    tbin = csim[rind[1:]] - csim[rind[:-1]]

    radial_prof = tbin / nr
    radial_prof = radial_prof[:int(np.floor(framew/2))-1]
    
    return(radial_prof)

def gen_azm_spatial_ft(filtered_stim, filt, raw_stim, stim_cpd, filt_name, fc):
    
    # Circular Averaged Power Spectrum - filtered_stim
    filtered_ft = azimuthalAverage(np.abs(np.fft.fftshift(np.fft.fft2(filtered_stim)))**2)
     # Circular Averaged Power Spectrum - raw_stim
    raw_ft = azimuthalAverage(np.abs(np.fft.fftshift(np.fft.fft2(raw_stim)))**2)
    # Circular Averaged Power Spectrum - filter
    filt_ft = azimuthalAverage(filt**2)
    
    #normalize all to max 1
    filtered_ft /= np.max(filtered_ft)
    raw_ft /= np.max(raw_ft)
    filt_ft /= np.max(filt_ft)
    
    # Get the Frequencies in CPD
    cpds = np.linspace(0,stim_cpd,len(raw_ft))

    #plot them all
    azm_plt = plt.figure(figsize = (8,6))
    azm_plt = plt.loglog(cpds, filtered_ft,  '.', 
                         label='filtered_stim')
    azm_plt = plt.loglog(cpds, raw_ft, '.', label='raw_stim')
    azm_plt = plt.loglog(cpds, filt_ft, '.', label='filter')
    #azm_plt = plt.plot(cpds, filtered_ft, label='filtered_stim',)
    #azm_plt = plt.plot(cpds, raw_ft, label='raw_stim')
    #azm_plt = plt.plot(cpds, filt_ft, label='filter')
    
    azm_plt = plt.axvline(fc, c='k')
    plt.xlabel('Frequency (cycles/deg)')
    plt.ylabel('Power')
    plt.title(f'Stim Fourier Spectra - {filt_name} Filter: Fc={fc}')
    plt.legend()
    
    return(azm_plt)

def gen_temporal_ft(filtered_stim, filt, raw_stim, stim_fps, filt_name, fc):
    
    # Circular Averaged Power Spectrum - filtered_stim
    filtered_ft = np.mean(np.abs(np.fft.fftshift(np.fft.fft(filtered_stim,axis=2)))**2,axis=(0,1))
     # Circular Averaged Power Spectrum - raw_stim
    raw_ft =  np.mean(np.abs(np.fft.fftshift(np.fft.fft(raw_stim,axis=2)))**2,axis=(0,1))
    # Circular Averaged Power Spectrum - filter
    filt_ft = filt**2
    
    #normalize all to max 1
    filtered_ft /= np.max(filtered_ft)
    raw_ft /= np.max(raw_ft)
    filt_ft /= np.max(filt_ft)
    
    # Get the Frequencies in fps
    fps_fqs = np.linspace(0,stim_fps,len(raw_ft))

    #plot them all
    fq_plt = plt.figure(figsize = (8,6))
    fq_plt = plt.loglog(fps_fqs, filtered_ft,  '.', 
                         label='filtered_stim')
    print(len(fps_fqs))
    fq_plt = plt.loglog(fps_fqs, raw_ft, '.', label='raw_stim')
    fq_plt = plt.loglog(fps_fqs, filt_ft, '.', label='filter')
    #fq_plt = plt.plot(cpds, filtered_ft, label='filtered_stim',)
    #fq_plt = plt.plot(cpds, raw_ft, label='raw_stim')
    #fq_plt = plt.plot(cpds, filt_ft, label='filter')
    
    fq_plt = plt.axvline(fc, c='k')
    plt.xlabel('Frequency (frames/sec)')
    plt.ylabel('Power')
    plt.title(f'Stim Fourier Spectra - {filt_name} Filter: Fc={fc}')
    plt.legend()
    
    return(azm_plt)


def filt_cosine_step(f, fc):
    #cosine step taper, full power at fd, zero power at 2*fd
    warn_flag = False
    #calc fd (taper start) and fz (taper end)
    fd = fc*np.pi/(np.arccos(np.sqrt(2)-1)+np.pi)
    #end taper at 2*fd for a power scale
    fz = 2*fd
    #can now define function
    filt = 0.5*(1+np.cos(np.pi*(f-fd)/(fz-fd+0.01)))

    filt[f < fd] = 1
    filt[f > fz] = 0

    if fz > np.max(f):
        warn_flag = True
        warnings.warn('Zero point is beyond Nyquist - Nothing is completally cuttoff.')
    return(filt, warn_flag)

def filt_gauss_step(f, fc):
    #cuttoff amplitude for frequencies above cuttoff based on gernalized gaussian
    warn_flag=False
    beta=2 #gaussian for now (beta=1 for Laplacian)
    alpha=0.05
    #calculate half width at half max: relationship between cuttoff fq and frequency where gauss taper is centerd.
    hwhm = alpha * (np.log(2))**(1./beta)
    #center of gaussian is cuttoff minus half witch half max
    fd = fc - hwhm
    #check if top of gaussian is negative.
    if fd < 0:
        warnings.warn('Taper Top Negative - Won\'t reach full contrast.')
        warn_flag = True
    #generic gauusian scaing function
    filt = np.exp(-1*(np.abs(f-fd)/alpha)**beta)
    filt /= np.max(filt)
    filt[f<fd] = 1.
    return(filt, warn_flag)


def fft_lowpass(img_in, cpd_cutoff, stim_cpd, filt_name='sharp', rescale=True):
    '''
    Lowpass filter an image at a given cpd cuttoff using Fourier representation, for a given cpd of the img stimulus
    
    Args:
        img_in (2d numpy array):   stimluius img
        cpd_cuttoff (float):  maximum CPD value present in output img
        stim_cpd (float):    CPD of stimulus (should be larger than cpd_cuttoff)
        filt_name (str):        define the type of filtering desired (sharp, cosine_step, gauss_step, gauss_taper)
        
    Returns:
        stim (2d numpy array):   stimlulus image fourier filtered and no frequencies higher than cpd_cuttoff
        mag (2d numpy array):   magnitude of stimulus
        phase (2d numpy array): global phase angle of stimulus
        filt (2d numpy array): fiter used to create filtered img
        warn_flag (bool):   flag  if we had a warning in generating image
    '''
    
    # warn flag is false by default
    warn_flag = False
    #make sure parameters make sense
    if cpd_cutoff > stim_cpd:
        warnings.warn('Cutoff CPD is higher than stimulus CPD')
        warn_flag=True

    #find ratio of cuttoff to max cpd so we know where to stop in fourier space
    fft_diameter_fc = cpd_cutoff/(stim_cpd)
    fft = np.fft.fftshift(np.fft.fft2(img_in))
    mag = np.abs(fft)
    xx, yy = np.meshgrid(np.linspace(-1, 1, img_in.shape[0]),
                         np.linspace(-1, 1, img_in.shape[1])) #THIS IS NOT VALID IF STIM IS NOT SQUARE111
    fft_diameters = np.sqrt(xx**2 + yy**2)
    
    #calculate filter
    if(filt_name=='sharp'):
    # Anything greater than cpd_cutoff set to 0 in the mag
        filt = (fft_diameters <= fft_diameter_fc)
        
    elif(filt_name=='gauss_taper'):
        #calculate sigma needed for HWHM value to be at fc
        #sigma = fft_diameter_fc / (np.sqrt(-np.log(np.sqrt(0.5))))
        #rescale to 100 so we don't get numerical instability
        sigma = fft_diameter_fc*100000 / np.sqrt(-2*np.log(np.sqrt(0.5)))
        filt = np.exp(-1.5*((fft_diameters*100000)/(sigma))**2)
    elif(filt_name=='cosine_step'):
        filt, warn_flag = filt_cosine_step(fft_diameters, fft_diameter_fc)
    
    elif(filt_name=='gauss_step'):
        filt, warn_flag = filt_gauss_step(fft_diameters, fft_diameter_fc)
    
    else:
        raise ValueError(f'{filt_name} is an unknown filtering type! Currently Supported Decompositions are \'fourier_sharp\', \'fourier_gauss\' and \'wavelet\'.  Returning original Signal.')
        filt = 1
        
    #multiply by arbitrary fourier filter
    mag = np.multiply(mag,filt)
    phase = np.angle(fft)
    
    # Reconstruct the image using the inverse fourier transform
    ifft = np.fft.fftshift(mag * np.exp(phase * 1.0j))
    img_ifft = np.fft.ifft2(ifft).real
    
    # Rescale to [0,255]
    if(rescale):
        img_ifft = imtools.rescale_255(img_ifft)
    
    return img_ifft, mag, phase, filt, warn_flag