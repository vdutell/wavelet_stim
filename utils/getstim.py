import numpy as np
def step_stim(width_px, height_px, stepdn=False, orient=1, contrast=1):
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
    #contrast
    stim = stim*contrast
            
    return(stim)
    