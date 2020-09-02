from root.utils import resize, setPercentileGrayThresholds, otsuBinarization, binarize, normalize
import numpy as np
import copy
from skimage.color import rgb2yiq
from skimage.morphology import closing, dilation
from skimage.filters import threshold_multiotsu
from skimage.morphology import square
from skimage.restoration import inpaint

have_hairs = [16, 18, 28, 46, 55, 58, 59, 60, 64, 65, 72, 74, 78, 95, 97, 99, 100, 111, 113, 118]
have_hairs_names = ['ISIC_9174306']
best_ones = [18, 58, 74, 118]
output_shape = (128, 128)

def hairs_remove(image, returnAllSteps= False):
    img = copy.copy(image)
    img = resize(img, output_shape)
    
    B_image = img[:, :, 2]

    #color space transformation RGB -> Y
    yiq = rgb2yiq(img)
    Y_image = yiq[:, :, 0]

    #hairs detection
    closed = closing(Y_image, square(5))
    hairs_mask = Y_image - closed
    t = threshold_multiotsu(hairs_mask, 4)[2]
    binary_hairs_mask = otsuBinarization(hairs_mask, t)
    inverted_binary_hairs_mask = binarize(binary_hairs_mask, 0)
    dilated_hairs_mask = dilation(inverted_binary_hairs_mask)

    #image impainting
    image_result = inpaint.inpaint_biharmonic(B_image, dilated_hairs_mask)

    if returnAllSteps:
        B_image[np.where(dilated_hairs_mask==1)] = 0
        hairs_mask_display = binarize(dilated_hairs_mask, 0)
        return image[:, :, 2], closed, hairs_mask, binary_hairs_mask, hairs_mask_display, B_image, normalize(image_result)
    return normalize(image_result)

'''    
'''