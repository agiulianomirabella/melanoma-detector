from root.utils import * # pylint: disable= unused-wildcard-import
from skimage.segmentation import flood_fill

shape = (128, 128)

x, y = np.indices(shape)
x_center, y_center = shape[0]/2, shape[1]/2

r = shape[0]/2 + 1
circle_mask = (x - x_center)**2 + (y - y_center)**2 > r**2

second_r = shape[0]/2
second_circle_mask = (x - x_center)**2 + (y - y_center)**2 > second_r**2


def segment(image):

    grayScale = toGray(image)
    grayScale[circle_mask] = 1

    grayScale = setPercentileGrayThresholds(grayScale, 5, 95)

    info = np.where(grayScale == np.min(grayScale))


    center = (grayScale.shape[0]//2, grayScale.shape[1]//2)


    x = [abs(e - center[0]) for e in info[0]]
    y = [abs(e - center[1]) for e in info[1]]

    z = np.array(x) + np.array(y)

    index = np.where(z == np.min(z))[0][0]

    seed_point = (center[0] + x[index], center[1] + y[index])

    print(np.min(grayScale))
    print(center)
    print(seed_point)
    print(grayScale[seed_point])

    grayScale = digitizeToEqualWidth(grayScale, 10)

    segmented = flood_fill(grayScale, seed_point, 1, connectivity=2, tolerance=0.2)
    segmented[seed_point[0]-2:seed_point[0]+2, seed_point[1]-2:seed_point[1]+2] = 3

    return grayScale, segmented




