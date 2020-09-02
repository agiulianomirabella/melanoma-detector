from root.utils import * # pylint: disable= unused-wildcard-import
from root.seg.hr import have_hairs, hairs_remove, best_ones

import os


image = readDicomImage('../data/input/train/' + os.listdir('../data/input/train')[have_hairs[4]])

image = resize(image, (128, 128))

grayScale, closed, hairs_mask, binary_hairs_mask, hairs_mask_display, B_image, image_result = hairs_remove(image, returnAllSteps=True)

displaySeveralImages([image, grayScale, closed, hairs_mask, image_result],
        titles=['original', 'grayScale', 'closed', 'hairs mask', 'result'])
