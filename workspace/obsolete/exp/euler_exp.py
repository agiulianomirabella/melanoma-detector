from root.utils import * # pylint: disable= unused-wildcard-import
from root.readData import readData
from root.topology.euler import eulerInfo
from root.segmentation.hr import hairs_remove

n = 4

images = getFirstNImages([n])
image = hairs_remove(images[0])

print(eulerInfo(image))
image = setPercentileGrayThresholds(image, 0.2, 99.8)
image = digitizeToEqualWidth(image, 5)
displayImage(image)

