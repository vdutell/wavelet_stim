import numpy as np

import utils.imtools as imtools
import utils.fouriertools as ftools
import utils.wavelettools as wtools
import utils.imwritetools as imwtools


def step_stim(width_px, height_px, stepdn=False, rescale=True, orient=1, contrast=1):
    '''
    Make a step function stimulus of a given size, orientation, and contrast
    
    Args:
        width_px (int):   width in pixels of stimulus
        height_px (int):  height in pixels of stimulus
        stepdn (bool):    L-R; U-D; go from white to black (down) vs black to white (up)
        orient (int):    orientation, 1=vertical, 0=horizontal
        contrast (float): float value between 0 and 1 of contrast for stimlulus
        
    Returns:
        stim (2d float):    step function stimulus  with values in [0,1]
    '''
    
    #vertical line step function
    if(orient==1):
        stim = np.hstack((np.zeros((height_px, width_px//2)),
                           np.ones((height_px, width_px//2))))
        if(stepdn):
            stim = stim[:,::-1]
    
    #horizontal line step function
    elif(orient==0):
        stim = np.vstack((np.zeros((height_px//2, width_px)),
                   np.ones((height_px//2, width_px))))
        if(stepdn):
            stim = stim[::-1,:]
    
    # Rescale to [0,255]
    if(rescale):
        stim = imtools.rescale_255(stim)
    
    #contrast
    stim = stim*contrast
            
    return(stim)
    
    
    import utils.fouriertools as ftools
import utils.wavelettools as wtools


def generate_stepfun_stims(stimpx_w, stimpx_h, stimdeg, cutoffs, filt='fourier_sharp', vertical=True):
    
    '''
    Generate filtered stepfun stimlui at the given cuttoffos with a given filter
    mode (See PIL modes: https://pillow.readthedocs.io/en/3.1.x/handbook/concepts.html#modes)
    '''
    
    outfolder = 'filtered_stims/'
    
    #calc degrees and cpd
    stim_cpd = (stimpx_w/2)/stimdeg
    
    stim_step = step_stim(stimpx_w, stimpx_h, orient=1, stepdn=True)
    
    #save our raw stim
    stim_fname = f'{outfolder}stepfun_raw_{stim_cpd}cpd.png'
    imwtools.writestim(stim_step, stim_fname)
    print(f'Wrote {stim_fname}')

    for idx, cut in enumerate(cutoffs):
        if(filt=='sharp'):
            stim_filt = ftools.fft_lowpass(stim_step, cut, stim_cpd, filt)[0] 
        elif(filt=='cosine_step'):
            stim_filt = ftools.fft_lowpass(stim_step, cut, stim_cpd, filt)[0]      
        elif(filt=='gauss_step'):
            stim_filt = ftools.fft_lowpass(stim_step, cut, stim_cpd, filt)[0] 
        elif(filt=='gauss_taper'):
            stim_filt = ftools.fft_lowpass(stim_step, cut, stim_cpd, filt)[0] 
        else:
            raise ValueError('Unknown filtering type! Currently Supported Decompositions are \'fourier_sharp\', \'fourier_gauss\' and \'wavelet\'')
        
        stim_fname = f'{outfolder}stepfun_{filt}_{cut}cpd.png'
        imwtools.writestim(stim_filt, stim_fname)
        print(f'Wrote {stim_fname}')
        
    return(stim_filt)

   