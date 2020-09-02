from root.utils import * # pylint: disable= unused-wildcard-import
from root.readData import readData
from root.segmentation.hr import hairs_remove
from root.segmentation.circle import circleCorrection

df, images, target = readData(2, 'train')
without_hairs = hairs_remove(images[0])

corrected = circleCorrection(without_hairs)

displaySeveralImages([images[0], without_hairs, corrected], titles=['original', 'without_hairs', 'corrected'])

