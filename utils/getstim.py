import numpy as np

import utils.imtools as imtools
import utils.fouriertools as ftools
import utils.wavelettools as wtools
import utils.imwritetools as imwtools
import pathlib

def step_stim_img(width_px, height_px, loc=0.5, stepdn=False, rescale=True, orient=1, contrast=1):
    '''
    Make a step function stimulus of a given size, orientation, and contrast
    
    Args:
        width_px (int):   width in pixels of stimulus
        height_px (int):  height in pixels of stimulus
        loc (float):      location of split from 0 (left/top) to 1 (right/bottom)
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


def step_stim(width_px, height_px, len_frames=1, stepdn=False, rescale=True, orient=1, contrast=1, reverse_phase=True):
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
    
    # if our step function is just a 1 frame image, return the image
    if(len_frames==1):
        stim = step_stim_img(width_px, height_px, loc=0.5, stepdn=False, rescale=True, orient=1, contrast=1)
    # otherwise, loop through the number of frames and make a moving edge.
    else:
        stim = np.zeros((len_frames, width_px, height_px))
        
        #if we are doing reverse_phase grating: make every other frame reverse.
        if(reverse_phase):
            stim[::2,:,:] = step_stim_img(width_px, height_px, loc=0.5, stepdn=False, rescale=True, orient=1, contrast=1)
            stim[1::2,:,:] = step_stim_img(width_px, height_px, loc=0.5, stepdn=True, rescale=True, orient=1, contrast=1)
            
            
    return(stim)



def generate_spatial_filtered_stims(stim, stimdeg, cutoffs, filt='fourier_sharp', stim_type='stepfun'):
    
    '''
    Generate filtered stimlui at the given cuttoffos with a given filter
    mode (See PIL modes: https://pillow.readthedocs.io/en/3.1.x/handbook/concepts.html#modes)
    '''
    
    stim_outfolder = 'filtered_stims/spatial/'
    ft_outfolder = stim_outfolder+'fts/'
    
    stimpx_w, stimpx_h = np.shape(stim)
    
    #calc degrees and cpd
    stim_cpd = (stimpx_w/2)/stimdeg
    
    #save our raw stim
    stim_fname = f'{stim_outfolder}{stim_type}_raw_{int(stim_cpd)}cpd.png'
    imwtools.writestim(stim, stim_fname)
    # create our raw stim's FT
    stim_ft = ftools.gen_azm_spatial_ft(stim, np.ones_like(stim), stim, stim_cpd, filt, int(stim_cpd))
    #save our raw stim's ft
    stim_ft_fname = f'{ft_outfolder}{stim_type}_raw_{int(stim_cpd)}cpd_ft.png'
    imwtools.writeplot(stim_ft, stim_ft_fname)
    print(f'Wrote {stim_ft_fname}')

    # loop through cuttoff frequencies and filter
    for idx, cut in enumerate(cutoffs):
        filt_stim, stim_mag, stim_phase, stim_filter, warn_flag = ftools.fft_lowpass(stim, cut, stim_cpd, filt)
        stim_filt_ft = ftools.gen_azm_spatial_ft(filt_stim, stim_filter, stim, stim_cpd, filt, cut)
        # if we had a warning during generating the image, reflect in image filename
        if(warn_flag):
            stim_fname = f'{stim_outfolder}{stim_type}_{filt}_{int(cut)}cpd_warn.png'
            stim_ft_fname = f'{ft_outfolder}{stim_type}{filt}_{int(cut)}cpd_ft_warn.png'
        else:
            stim_fname = f'{stim_outfolder}{stim_type}_{filt}_{int(cut)}cpd.png'
            stim_ft_fname = f'{ft_outfolder}{stim_type}_{filt}_{int(cut)}cpd_ft.png'
        # write to disk
        imwtools.writestim(filt_stim, stim_fname)
        imwtools.writeplot(stim_filt_ft, stim_ft_fname)
        print(f'Wrote {stim_fname}; {stim_ft_fname}')
        
    return(filt_stim)

   
    
def generate_temporal_filtered_stims(stim, stimfps, cutoffs, filt='fourier_sharp', stim_type='stepfun'):
    
    '''
    Generate filtered stimlui at the given cuttoffos with a given filter
    mode (See PIL modes: https://pillow.readthedocs.io/en/3.1.x/handbook/concepts.html#modes)
    '''
    
    stim_outfolder = 'filtered_stims/temporal/'
    ft_outfolder = stim_outfolder+'fts/'
    
    stimlen_frames, stimpx_w, stimpx_h = np.shape(stim)
    
    #calc degrees and cpd
    #stim_cpd = (stimpx_w/2)/stimdeg
    
    
    #save our raw stim
    stim_dir = f'{stim_outfolder}{stim_type}_raw_{int(stimfps)}fps/'
    pathlib.Path(stim_dir).mkdir(exist_ok=True)
    for i, frame in enumerate(stim):
        stim_fname = stim_dir + f'frame_{i+1}.png'
        imwtools.writestim(frame, stim_fname)
    # create our raw stim's FT
    stim_ft = ftools.gen_temporal_ft(stim, np.ones_like(stim), stim, stimfps, filt, int(stimfps))
    #save our raw stim's ft
    stim_ft_fname = f'{ft_outfolder}{stim_type}_raw_{int(stimfps)}fps_ft.png'
#    imwtools.writeplot(stim_ft, stim_ft_fname)
#    print(f'Wrote {stim_ft_fname}')

    # loop through cuttoff frequencies and filter
    for idx, cut in enumerate(cutoffs):
        filt_stim, stim_mag, stim_phase, stim_filter, warn_flag = ftools.fft_temporal_lowpass(stim, cut, stim_cpd, filt)
#        stim_filt_ft = ftools.gen_azm_ft(filt_stim, stim_filter, stim, stim_cpd, filt, cut)
#        # if we had a warning during generating the image, reflect in image filename
##        if(warn_flag):
#            stim_fname = f'{stim_outfolder}{stim_type}_{filt}_{int(cut)}cpd_warn.png'
#            stim_ft_fname = f'{ft_outfolder}{stim_type}{filt}_{int(cut)}cpd_ft_warn.png'
##        else:
 #           stim_fname = f'{stim_outfolder}{stim_type}_{filt}_{int(cut)}cpd.png'
 #           stim_ft_fname = f'{ft_outfolder}{stim_type}_{filt}_{int(cut)}cpd_ft.png'
 #       # write to disk
        #save our fitlered stim
    stim_dir = f'{stim_outfolder}{stim_type}_{filt}_{int(cut)}fps/'
    pathlib.Path(stim_dir).mkdir(exist_ok=True)
    for i, frame in enumerate(filt_stim):
        stim_fname = rawstim_dir + f'frame_{i+1}.png'
        imwtools.writestim(frame, stim_fname)
 ##       imwtools.writeplot(stim_filt_ft, stim_ft_fname)
  #      print(f'Wrote {stim_fname}; {stim_ft_fname}')
        
  #  return(filt_stim)