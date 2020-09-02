from root.utils import * # pylint: disable= unused-wildcard-import

import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
from skimage.feature import canny
from skimage.filters import threshold_otsu, threshold_triangle, threshold_isodata
from skimage.transform import resize
from skimage.exposure import equalize_hist

shape = (256, 256)

x, y = np.indices(shape)
x_center, y_center = shape[0]/2, shape[1]/2

r = shape[0]/2 + 1
circle_mask = (x - x_center)**2 + (y - y_center)**2 > r**2

second_r = shape[0]/2
second_circle_mask = (x - x_center)**2 + (y - y_center)**2 > second_r**2


def segment(image, returnSegmentedImage= True, returnAllSteps= False):
    #return:
    #   - snake: (n_features, 2) shaped array
    #   - segmented image: 2d-array with initial contours points plotted in red and final ones in blue

    grayScale = toGray(image)
    grayScale[circle_mask] = 0

    center = [grayScale.shape[0]/2, grayScale.shape[1]/2]
    radius = grayScale.shape[0]*0.3

    s = np.linspace(0, 2*np.pi, 400)
    r = center[0] + radius*np.sin(s)
    c = center[1] + radius*np.cos(s)
    init = np.array([r, c]).T

    gauss_image = gaussian(grayScale, 3)

    snake = active_contour(gauss_image,
                        init, alpha=0.018, beta=0.1, gamma=0.001, w_edge=3,
                        w_line=0, coordinates='rc')

    '''
        def active_contour(image, snake, alpha=0.01, beta=0.1,
                        w_line=0, w_edge=1, gamma=0.01,
                        bc=None, max_px_move=1.0,
                        max_iterations=2500, convergence=0.1,
                        *,
                        boundary_condition='periodic',
                        coordinates=None):

        alpha : Snake length shape parameter. Higher values makes snake contract faster.
        beta  : Snake smoothness shape parameter. Higher values makes snake smoother.
        w_line: Controls attraction to brightness. Use negative values to attract toward dark regions.
        w_edge: Controls attraction to edges. Use negative values to repel snake from edges.
        gamma : Explicit time stepping parameter.
    '''

    if returnAllSteps:
        #Create the new image:
        fig, ax = plt.subplots(figsize=(14, 14))
        ax.imshow(gauss_image, cmap='gray')
        ax.plot(init[:, 1], init[:, 0], '--r', lw=3) # pylint: disable=unsubscriptable-object
        ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, grayScale.shape[1], grayScale.shape[0], 0])

        fig.canvas.draw()
        
        #Save the segmented canny_image into a numpy array:
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        segmented_gauss_image = resize(data, (512, 512))

        #Create the new image:
        fig, ax = plt.subplots(figsize=(14, 14))
        ax.imshow(image, cmap='gray')
        ax.plot(init[:, 1], init[:, 0], '--r', lw=3) # pylint: disable=unsubscriptable-object
        ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, grayScale.shape[1], grayScale.shape[0], 0])

        fig.canvas.draw()
        
        #Save the segmented image into a numpy array:
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        segmented_image = resize(data, (512, 512))

        return snake, grayScale, gauss_image, segmented_gauss_image, segmented_image

    if returnSegmentedImage:
        fig, ax = plt.subplots(figsize=(14, 14))
        ax.imshow(image, cmap='gray')
        ax.plot(init[:, 1], init[:, 0], '--r', lw=3) # pylint: disable=unsubscriptable-object
        ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, grayScale.shape[1], grayScale.shape[0], 0])

        fig.canvas.draw()
        
        #Save the segmented image into a numpy array:
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        segmented_image = resize(data, (512, 512))

        return snake, segmented_image

    else:
        return snake



