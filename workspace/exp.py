from root.utils import * # pylint: disable= unused-wildcard-import

import skimage.io as io

image = io.imread('../data/halfway/hr/128/ISIC_0599047_hr.jpg')
digitized = digitizeToEqualWidth(image, 5)
displaySeveralImages([image, digitized])
