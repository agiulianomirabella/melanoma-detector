from root.utils import * # pylint: disable= unused-wildcard-import
from ripser import ripser
import persim
from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from root.segmentation.active_contours import segment
from scipy.ndimage import find_objects
from skimage.transform import resize

def extractRegion(image):

    contours = np.zeros(image.shape[:2])
    snake = segment(image, returnSegmentedImage= False)

    for c in snake:
        contours[tuple(c.astype(int))]=1

    margins = find_objects(contours.astype(int))[0]
    x = margins[0]
    y = margins[1]
    newX = slice(x.start - 10, x.stop + 10)
    newY = slice(y.start - 10, y.stop + 10)
    
    return rgb2gray(image[newX, newY])

def extractPoints(image):
    img = image.copy()

    region = extractRegion(img)

    region = applyOtsuThreshold(region, threshold_otsu(region))
    points = getGrayValueCoordinates(region, 0)

    return np.array(points)

def display_persistent_diagram(data):
    #data: (n_samples, 2) shaped numpy array
    dgms = ripser(data)['dgms']
    persim.plot_diagrams(dgms, show= True)

