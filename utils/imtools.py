import numpy as np
import skimage.transform
import warnings

# Crop and resize
def cropnresize(img, imsize):
    img_out = img[img.shape[0]//2-np.min(img.shape[:2])//2:img.shape[0]//2+
                  np.min(img.shape[:2])//2,
                  img.shape[1]//2-np.min(img.shape[:2])//2:img.shape[1]//2+
                  np.min(img.shape[:2])//2]
    if img.shape != imsize:
        img_out = skimage.transform.resize(img_out, imsize, mode='reflect')
        warnings.warn(f'Cropped and resized image from {img.shape} to {img_out.shape}')
    return img_out
    
    # Convert to grayscale
def rgb2gray(rgb):
    # Convert to grayscale: https://stackoverflow.com/a/12201744
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])
