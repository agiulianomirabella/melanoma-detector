from root.utils import * # pylint: disable= unused-wildcard-import
from skimage.filters import threshold_otsu, threshold_multiotsu
from scipy import ndimage as ndi
import math
import copy

def brightnessCorrection(image):
    # Generate a circle mask to correct brightness misinformation

    shape = image.shape
    n = math.ceil(shape[0]*(math.sqrt(2)-1)/2)
    if n%2 != 0:
        n = n+1
    n = int(n/2)

    augmented_shape = (shape[0]+n, shape[1]+n)

    x, y = np.indices(augmented_shape)
    x_center, y_center = augmented_shape[0]/2, augmented_shape[1]/2
    r = augmented_shape[0]/2
    circle_image = (x - x_center)**2 + (y - y_center)**2 < r**2

    brightness_mask = ndi.distance_transform_edt(circle_image)
    brightness_mask = brightness_mask[n-1:shape[0]+n-1, n-1:shape[1]+n-1]

    print()
    print(shape)
    print(augmented_shape)
    print(n)
    print(shape[0]+n)
    print(shape[1]+n)
    print()

    brightness_mask = np.zeros(shape) - brightness_mask
    brightness_mask = normalize(brightness_mask) + 1

    out = np.multiply(image, brightness_mask)

    return brightness_mask, out

