
"""
DHR tasks:
    
    -- Applying Morphological Black-Hat transformation
    -- Creating the mask for InPainting task
    -- Applying inpainting algorithm on the image

"""

import cv2
import copy
import numpy as np
from skimage.color import rgb2gray

def hairs_remove2(image):
    # Convert the original image to grayscale
    img = copy.copy(image)
    grayScale = rgb2gray(img)

    # Kernel for the morphological filtering
    kernel = cv2.getStructuringElement(1, (41,41))

    # Perform the blackHat filtering on the grayscale image to find the 
    # hair countours
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

    # intensify the hair countours in preparation for the inpainting 
    # algorithm
    ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    # inpaint the original image depending on the mask
    grayScale32 = np.float32(grayScale)
    thresh2 = thresh2.astype(np.uint8)
    dst = cv2.inpaint(grayScale32, thresh2, 9, cv2.INPAINT_TELEA)

    return dst

