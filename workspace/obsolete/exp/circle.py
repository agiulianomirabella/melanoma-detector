from root.utils import * # pylint: disable= unused-wildcard-import
from skimage.filters import threshold_otsu, threshold_multiotsu
from scipy import ndimage as ndi
import math
import copy

shape = (128, 128)

x, y = np.indices(shape)
x_center, y_center = shape[0]/2, shape[1]/2
r = shape[0]/2
circle_mask = (x - x_center)**2 + (y - y_center)**2 > r**2

def circleCorrection(image):
    # Generate a circle mask to correct brightness misinformation

    out = copy.copy(image)

    out[circle_mask]=0

    return out

