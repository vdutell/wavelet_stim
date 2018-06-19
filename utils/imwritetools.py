import numpy as np
from PIL import Image 
import matplotlib.pyplot as plt

import utils.imtools as imtools

def writestim(img, fname, rescale=True, mode='L'):
    
    im = Image.fromarray(img).convert('RGB')
    im.save(fname, "PNG")