from root.utils import * # pylint: disable= unused-wildcard-import
import numpy as np
from root.topology.tani_utils import euler

number_of_grays = 5

def eulerInfo(image):

    out = []

    img = setPercentileGrayThresholds(image, 0.2, 99.8)

    digitized = digitizeToEqualWidth(img, number_of_grays)
    gray_values = np.linspace(0, 1, number_of_grays)

    for g in gray_values:
        coordinates = getGrayValueCoordinates(digitized, g)
        out.append(euler(coordinates))

    return out


