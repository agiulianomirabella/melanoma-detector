from root.utils import * # pylint: disable= unused-wildcard-import
from skimage.segmentation import chan_vese
from scipy.ndimage import binary_opening

shape = (128, 128)
x, y = np.indices(shape)
x_center, y_center = shape[0]/2, shape[1]/2

r = shape[0]/2 - 10
circle_mask = (x - x_center)**2 + (y - y_center)**2 > r**2

def segment(image, mu=0.15, lambda1=1, lambda2=1, tol=1e-3, max_iter=500, dt=0.5, extended_output=False):

    grayScale = image.copy()
    grayScale[circle_mask]=0

    segmented, phi, energies = chan_vese(grayScale, mu, lambda1, lambda2, tol, max_iter,
                dt, init_level_set="checkerboard", extended_output=True)

    segmented[circle_mask]=1

    out = np.zeros(segmented.shape)
    out[segmented==0]=1

    out = binary_opening(out).astype(int)

    if extended_output:
        return grayScale, out, 
    else:
        return out


    