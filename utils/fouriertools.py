import warnings
import numpy as np
import math

import utils.imtools as imtools

import matplotlib as mpl
import matplotlib.pyplot as plt

import matplotlib.gridspec as gsp

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
    
    # Power Spectrum - filtered_stim
    filtered_ft = np.mean(np.abs(np.fft.fftshift(np.fft.fft(filtered_stim,axis=0)))**2,axis=(1,2))
    filtered_ft = filtered_ft[len(filtered_ft)//2:]
    # Power Spectrum - raw_stim
    raw_ft =  np.mean(np.abs(np.fft.fftshift(np.fft.fft(raw_stim,axis=0)))**2,axis=(1,2))
    raw_ft = raw_ft[len(raw_ft)//2:]
    # Circular Averaged Power Spectrum - filter
    filt_ft = np.mean(filt**2,axis=(1,2))
    filt_ft = filt_ft[len(filt_ft)//2:]

    #normalize all to max 1
    filtered_ft /= np.max(filtered_ft)
    raw_ft /= np.max(raw_ft)
    filt_ft /= np.max(filt_ft)
    
    # Get the Frequencies in fps (0 to nyquist)
    fps_fqs = np.linspace(0,stim_fps/2,len(raw_ft))
    
    #print('fqs:')
    #print(fps_fqs)
    #print('filtered:')
    #print(filtered_ft)
    #print('filter:')
    #print(filt_ft)#, raw_ft)

    #plot them all
    fq_plt = plt.figure(figsize = (8,6))
    fq_plt = plt.loglog(fps_fqs, filtered_ft,  '.', label='filtered_stim')
    fq_plt = plt.loglog(fps_fqs, raw_ft, '.', label='raw_stim')
    fq_plt = plt.loglog(fps_fqs, filt_ft, '.', label='filter')
    
    #fq_plt = plt.figure(figsize = (8,6))
    #fq_plt = plt.plot(fps_fqs, filtered_ft,  '.', label='filtered_stim')
    #fq_plt = plt.plot(fps_fqs, raw_ft, '.', label='raw_stim')
    #fq_plt = plt.plot(fps_fqs, filt_ft, '.', label='filter')
    
    fq_plt = plt.axvline(fc, c='k')
    plt.xlabel('Frequency (frames/sec)')
    plt.ylabel('Power')
    plt.title(f'Stim Fourier Spectra - {filt_name} Filter: Fc={fc}')
    plt.legend()
    
    return(fq_plt)


def gen_spatialtemporal_ft(filtered_stim, filt, raw_stim, stim_cpd, stimfps, filt_name, s_fc, t_fc):
    
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
    plt.title(f'Stim Fourier Spectra - {filt_name} Filter: Fc_  ={fc}')
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

def fft_temporal_lowpass(stim_in, fps_cutoff, stim_fps, filt_name='sharp', rescale=True):
    '''
    Lowpass filter an movie at a given frequency cuttoff using Fourier representation, for a given fps of the stimulus
    
    Args:
        stim_in (3d numpy array):   stimluius movie (time, x, y)
        fps_cuttoff (float):  maximum fps value present in output img
        stim_fps (float):    FPS of stimulus (should be larger than fps_cuttoff)
        filt_name (str):     define the type of filtering desired (sharp, cosine_step, gauss_step, gauss_taper)
        
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
    if fps_cutoff > stim_fps:
        warnings.warn('Cutoff FPS is higher than stimulus FPS')
        warn_flag=True
    #size of stim
    stimf, stimx, stimy = np.shape(stim_in)
        
    #find ratio of cuttoff to max cpd so we know where to stop in fourier space
    fft_len_fc = fps_cutoff/(stim_fps/2)
    fft = np.fft.fftshift(np.fft.fft(stim_in,axis=0))
    mag = np.abs(fft)
    fft_diameters = np.tile(np.abs(np.linspace(-1,1,stimf)),(stimx, stimy, 1)).T
    #print(fft_diameters[:,1,1])
    #print(fft_len_fc)
    
    #calculate filter
    if(filt_name=='sharp'):
    # Anything greater than cpd_cutoff set to 0 in the mag
        filt = (fft_diameters <= fft_len_fc) 
        
    elif(filt_name=='gauss_taper'):
        #calculate sigma needed for HWHM value to be at fc
        #sigma = fft_diameter_fc / (np.sqrt(-np.log(np.sqrt(0.5))))
        #rescale to 100 so we don't get numerical instability
        sigma = fft_len_fc*100000 / np.sqrt(-2*np.log(np.sqrt(0.5)))
        filt = np.exp(-1.5*((fft_diameters*100000)/(sigma))**2)
    elif(filt_name=='cosine_step'):
        filt, warn_flag = filt_cosine_step(fft_diameters, fft_len_fc)
    
    elif(filt_name=='gauss_step'):
        filt, warn_flag = filt_gauss_step(fft_diameters, fft_len_fc)
    
    else:
        raise ValueError(f'{filt_name} is an unknown filtering type! Currently Supported Decompositions are \'fourier_sharp\', \'fourier_gauss\' and \'wavelet\'.  Returning original Signal.')
        filt = 1
        
    #multiply by arbitrary fourier filter
    mag = np.multiply(mag,filt)
    #print(mag[:,1,1])

    phase = np.angle(fft)
    
    # Reconstruct the movie using the inverse fourier transform
    ifft = np.fft.fftshift(mag * np.exp(phase * 1.0j))
    img_ifft = np.fft.ifft(ifft,axis=0).real
    
    # Rescale to [0,255]
    if(rescale):
        img_ifft = imtools.rescale_255(img_ifft)
    
    return img_ifft, mag, phase, filt, warn_flag

def fft_spatiotemporal_lowpass(stim_in, cpd_cutoff, fps_cutoff, stim_cpd, stim_fps, filt_name='sharp', rescale=True):
    '''
    Lowpass filter an movie at a given frequency cuttoff using Fourier representation, for a given fps of the stimulus
    
    Args:
        stim_in (3d numpy array):   stimluius movie (time, x, y)
        cpd_cutoff (float): maximum cpd value present in output stim
        fps_cutoff (float):  maximum fps value present in output stim
        stim_cpd (flot):    CPD of stimulus (should be larger than cpd_cuttoff)
        stim_fps (float):    FPS of stimulus (should be larger than fps_cuttoff)
        filt_name (str):     define the type of filtering desired (sharp, cosine_step, gauss_step, gauss_taper)
        
    Returns:
        stim (2d numpy array):   stimlulus image fourier filtered and no frequencies higher than cpd_cuttoff
        mag (2d numpy array):   magnitude of stimulus
        phase (2d numpy array): global phase angle of stimulus
        filt (2d numpy array): fiter used to create filtered img
        warn_flag (bool):   flag  if we had a warning in generating image
    '''
    
    # warn flags are false by default
    warn_flag_s = False
    warn_flag_t = False
    #make sure parameters make sense
    if fps_cutoff > stim_fps/2:
        warnings.warn('Cutoff FPS is higher than stimulus CPS')
        warn_flag_s=True
    #size of stim
    stimf, stimx, stimy = np.shape(stim_in)
        
    #find ratio of cuttoff to max cpd so we know where to stop in fourier space
    fft_len_fc_s = cpd_cutoff/(stim_cpd)
    #find same ratio in fps
    fft_len_fc_t = fps_cutoff/(stim_fps/2)
    #get ft of stim
    fft = np.fft.fftshift(np.fft.fftn(stim_in))
    mag = np.abs(fft)
    phase = np.angle(fft)
    #calculate diameters (frequencies) of signal
    xx, yy = np.meshgrid(np.linspace(-1, 1, stimx),
                         np.linspace(-1, 1, stimy)) #THIS IS NOT VALID IF STIM IS NOT SQUARE111
    fft_diameters_s = np.sqrt(xx**2 + yy**2)
    fft_diameters_t = np.tile(np.abs(np.linspace(-1,1,stimf)),(stimx, stimy, 1)).T
    
    #calculate filter
    if(filt_name=='sharp'):
    # Anything greater than cpd_cutoff set to 0 in the mag
        filt_s = (fft_diameters_s <= fft_len_fc_s) 
        filt_t = (fft_diameters_t <= fft_len_fc_t) 
        # pointwise multiply spatial and temporal filters to get full filter.
        filt = np.multiply(filt_s, filt_t)
        
    elif(filt_name=='gauss_taper'):
        #calculate sigma needed for HWHM value to be at fc
        #sigma = fft_diameter_fc / (np.sqrt(-np.log(np.sqrt(0.5))))
        #rescale to 100 so we don't get numerical instability
        sigma_s = fft_len_fc_s*100000 / np.sqrt(-2*np.log(np.sqrt(0.5)))
        filt_s = np.exp(-1.5*((fft_diameters_s*100000)/(sigma_s))**2)
        sigma_t = fft_len_fc_t*100000 / np.sqrt(-2*np.log(np.sqrt(0.5)))
        filt_t = np.exp(-1.5*((fft_diameters_s*100000)/(sigma_t))**2)
        # pointwise multiply spatial and temporal filters to get full filter.
        filt = np.multiply(filt_s, filt_t)
        
    elif(filt_name=='cosine_step'):
        filt_s, warn_flag_s = filt_cosine_step(fft_diameters_s, fft_len_fc_s)
        filt_t, warn_flag_t = filt_cosine_step(fft_diameters_t, fft_len_fc_t)
        # pointwise multiply spatial and temporal filters to get full filter.
        filt = np.multiply(filt_s, filt_t)
        
    elif(filt_name=='gauss_step'):
        filt_s, warn_flag_s = filt_gauss_step(fft_diameters_s, fft_len_fc_s)
        filt_t, warn_flag_t = filt_gauss_step(fft_diameters_t, fft_len_fc_t)
        # pointwise multiply spatial and temporal filters to get full filter.
        filt = np.multiply(filt_s, filt_t)
    
    else:
        raise ValueError(f'{filt_name} is an unknown filtering type! Currently Supported Decompositions are \'sharp\', \'gauss_taper\', \'cosine_step\' and \'gauss_step\'.  Returning original Signal.')
        filt = 1
        
    #multiply by arbitrary fourier filter
    mag = np.multiply(mag,filt)
    #print(mag[:,1,1])

    # Reconstruct the movie using the inverse fourier transform
    ifft = np.fft.fftshift(mag * np.exp(phase * 1.0j))
    stim_ifft = np.fft.ifftn(ifft).real
    
    # Rescale to [0,255]
    if(rescale):
        stim_ifft = imtools.rescale_255(stim_ifft)
        
    warn_flag = warn_flag_s | warn_flag_t
    
    return stim_ifft, mag, phase, filt, warn_flag


def make_3d_ft(movie, chunkshape, fps, ppd):
    '''
    Converts pixels from a trace to the corresponding pixels in a movie
    
    Parameters
    ----------
    movie:      3d numpy array definig movie for 3d fourier transform analysis.
    chunkshape:  3ple defining shape (frames,x,y) of movie 'chunks'
    
    Returns:
    --------
    mftchunk
    azmchunk
    freqspace1d
    freqspacefull
    freqtime
    
    '''
    
    #If movie is in color, average three color channels to get greyscale
    if(movie.ndim > 3):
        movie = np.mean(movie,axis=3)
    
    #movie shape
    #nframes = np.shape(movie)[0] #in frames
    #framew = np.shape(movie)[1] #in pixels
    #frameh = np.shape(movie)[2] #in pixels

    
    #if(chunklen%2 != 0 | chunklen<3):
    #    raise Exception('Error, chunklen must be an even number > 2!')
        
    #nchunks = int(np.floor(len(movie[:,0,0])/chunklen))
    
    #remove mean (DC component)
    #movie = movie - np.mean(movie)
    
    #take fourier transform over each fame
    #fttimecube = np.fft.fft2(movie)
    
    chunks = cubify(movie, chunkshape)
    del(movie)
    ftchunks = []
    
    #break into chunks
    for chunk in chunks:
        #chunkmov = movie[chunk*chunklen:(chunk+1)*chunklen]
        #remove mean (DC component)
        chunk = chunk - np.mean(chunk)
        #space FT for each chunk
        #chunklist.append(np.fft.fftn(chunkmov,axes=[0]))
        
        #3d ft for each chunk
        ftchunks.append(np.fft.fftn(chunk))
        
    del(chunks) #save space: we are done with raw chunks
    chunklen = len(ftchunks)
    
    ftchunks=np.array(ftchunks)
    
    print(f'ftchunks: {ftchunks.shape}')
    #take mean over all ftchunks to get one ftchunk
    mftchunk = np.mean(ftchunks,axis=0)
    #do fft shifts to make football
    mftchunk = np.fft.fftshift(mftchunk)
    print(f'mftchunk: {mftchunk.shape}')
    
    #array to hold azmaverage
    #azmchunk = np.zeros([chunklen,int(np.floor(framew/2))-1], dtype=complex)
    
    azmchunk = []
    ##spin to get mean
    for f in range(np.shape(mftchunk)[0]):
        azmchunk.append(azimuthalAverage(np.abs(mftchunk[f])))
    del(mftchunk)
    #only take positive side (real)
    azmchunk = np.abs(np.array(azmchunk[int(chunkshape[0]/2):]))
    print(f'azmchunk: {azmchunk.shape}')
        
    #azmchunk = (azmchunk[int(chunklen/2):] + azmchunk[int(chunklen/2):0:-1]) / 2
      
    #get the sampling rate for azmavgd
    freqspace1d = np.fft.fftfreq(chunkshape[1], d=1./ppd)[0:int(np.floor(chunkshape[1]/2))-1]
    #get the sampling rates
    freqspacefull = np.fft.fftshift(np.fft.fftfreq(chunkshape[1], d=1./ppd))
    freqtime = np.fft.fftshift(np.fft.fftfreq(chunkshape[0], d=1./fps))[int(chunkshape[0]/2):]
    
    #normalize the fft based on dx and dy
    dspace = freqspace1d[1] - freqspace1d[0]
    dtime = freqtime[1] - freqtime[0]
    #azmchunk = azmchunk - np.mean(azmchunk)
    azmchunk *= np.real(np.abs(dspace*dtime))
    
    
    return(azmchunk, freqspace1d, freqtime)


#def da_spatiotemporal_ft(amp_spectrum, fqspace, fqtime, nsamples = 7, figname='nameless', logscale=False):
    
def azm_avg_frames(stim):
    ft = []
    for i, frame in enumerate(stim):
        ft.append(azimuthalAverage(frame))
    ft = np.array(ft)

    #normalizeto max 1
    ft /= np.max(ft)
    return(ft)
    
    
def st_ft(stim):
    # Amplitude Spectrum - filtered_stim
    ft3d = np.abs(np.fft.fftshift(np.fft.fftn(stim)))
    ft = azm_avg_frames(ft3d)
    
    return(ft)



def da_spatiotemporal_ft(ft, stim_cpd, stim_fps, filt_name, fc_s, fc_t, save_fname, nsamples=5, logscale=False):
    
    figname = f'{filt_name}: Fc_s={fc_s}, Fc_t={fc_t}'
    
    #print(np.shape(ft)) #debugging
    ntfqs, nsfqs = np.shape(ft)
    
    # Get the frequencies in cpd
    fqspace= np.linspace(0,stim_cpd, nsfqs)
    # Get the Frequencies in fps (0 to nyquist)
    fqtime = np.linspace(0,stim_fps//2, ntfqs)
    
    #sampling indexes:
    space_start_sample = len(fqspace)//20
    space_end_sample = len(fqspace) - space_start_sample
    time_start_sample = len(fqtime)//20
    time_end_sample = len(fqtime) - time_start_sample
    
    #colors for lines
    spacesamplefqs_idx = np.linspace(space_start_sample,
                                     space_end_sample,
                                     nsamples).astype(int)
    timesamplefqs_idx = np.linspace(time_start_sample,
                                    time_end_sample,
                                    nsamples).astype(int)

    spacecolors = np.array(['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'])#[::-1]
    timecolors = spacecolors
    
    #make a grid
    fig = plt.figure(figsize=(10,6))
    full_grid = gsp.GridSpec(2,3)
    
    #layout of subplots
    grid_hm = gsp.GridSpecFromSubplotSpec(1,1,subplot_spec=full_grid[0:2,0:2])
    grid_time = gsp.GridSpecFromSubplotSpec(1,1,subplot_spec=full_grid[1,2])
    grid_space = gsp.GridSpecFromSubplotSpec(1,1,subplot_spec=full_grid[0,2])
    
    #heatmap
    axes_hm = plt.subplot(grid_hm[0])
    
    # plot the correct spectrum (power/amplitude/psd)
    # http://www.ldeo.columbia.edu/users/menke/research_notes/menke_research_note157.pdf
    joint_psd = 2/len(fqspace)*len(fqtime)*(np.abs(ft)**2)
    #joint_psd = 2/(len(fqspace)*len(fqtime)) * (np.abs(amp_spectrum)**2)
    plot_spectrum = joint_psd.T
    #convert to db
    plot_spectrum /= np.max(plot_spectrum) #normalize so max val is zero (DB scaling)
    plot_spectrum = 20*np.log10(plot_spectrum)

    hm = axes_hm.contourf(fqtime, fqspace, plot_spectrum,
                         cmap='gray')
    if(logscale):
        axes_hm.set_xscale("log") 
        axes_hm.set_yscale("log")
    axes_hm.set_xlabel('Hz')
    axes_hm.set_ylabel('cycles/deg')
    axes_hm.set_title(f'{figname} Log Power') 
    plt.colorbar(hm)

    #add lines
    for s in range(nsamples):
        #lines in time
        axes_hm.axvline(fqtime[timesamplefqs_idx[s]],c=timecolors[s],ls='-')
        #lines in space
        axes_hm.axhline(fqspace[spacesamplefqs_idx[s]],c=spacecolors[s],ls='--')

    #spaceplot
    axes_space = plt.subplot(grid_space[0])
    for i, tf_idx in enumerate(timesamplefqs_idx):
        axes_space.semilogx(fqspace, plot_spectrum[:,tf_idx],
                        label='{0:0.1f} Hz'.format(fqtime[tf_idx]),
                        c=timecolors[i])
    
    #axes_space.plot(fqspace[1:],1e-80/(fqspace[1:]),c='black') # 1/f line
    #axes_space.plot(fqspace[1:],1e-80/(fqspace[1:]**2),c='black') # 1/f^2 line
    
    axes_space.set_title('Spatial Frequency')
    axes_space.set_xlabel('cpd')
    axes_space.set_ylabel('Log Power')
    axes_space.set_xlim(fqspace[1],fqspace[-1])
    axes_space.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))
    axes_space.legend(fontsize=8)

    #timeplot
    axes_time = plt.subplot(grid_time[0])
    for i, sf_idx in enumerate(spacesamplefqs_idx):
        axes_time.semilogx(fqtime, plot_spectrum[sf_idx,:],
                       label='{0:0.1f} cpd'.format(fqspace[sf_idx]),
                       c=spacecolors[i])
    #axes_time.plot(fqtime[1:],1e4/(fqtime[1:]),c='black') # 1/f line
    #axes_time.plot(fqtime[1:],1e7/(fqtime[1:]**2),c='black') # 1/f^2 line
    axes_time.set_title('Temporal Frequency')
    axes_time.set_xlabel('Hz')
    axes_time.set_ylabel('Log Power') 
    axes_time.set_xlim(fqtime[1],fqtime[-1])
    axes_time.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%d'))
    axes_time.legend(fontsize=8)

    plt.tight_layout()
    
    plt.savefig(save_fname)