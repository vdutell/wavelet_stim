import numpy as np
from PIL import Image 

import utils.imtools as imtools

def writestim(img, fname, rescale=False, mode='L'):
    
    if(rescale):
        img = imtools.rescale(img)
    
    im = Image.new(mode, np.shape(img))
    im.putdata(img)
    im.save(fname)