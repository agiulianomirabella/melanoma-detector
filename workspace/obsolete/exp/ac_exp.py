from root.utils import * # pylint: disable= unused-wildcard-import
from root.readData import readData
from root.segmentation.active_contours import segment
from root.segmentation.hr import hairs_remove
import time

t1 = time.time()
train_df, train_images, train_labels, test_df, test_images, test_labels = readData(10)

images = train_images[:2]

print()
for i in train_df.image_name:
    print(i)
print()

t2 = time.time()
without_hairs = [hairs_remove(img) for img in images]

t3 = time.time()
outputs = [segment(img, returnAllSteps=True) for img in without_hairs]

grayScale             = [e[1] for e in outputs]
gauss_image           = [e[2] for e in outputs]
segmented_gauss_image = [e[3] for e in outputs]
segmented_image       = [e[4] for e in outputs]

first  = [images[0], without_hairs[0], grayScale[0], gauss_image[0], segmented_gauss_image[0], segmented_image[0]]
second  = [images[1], without_hairs[1], grayScale[1], gauss_image[1], segmented_gauss_image[1], segmented_image[1]]

t4 = time.time()

displaySeveralImages(first + second, ncols= 6, 
    titles=['original', 'without_hairs', 'grayScale', 'gauss_image', 'segmented_gauss_image', 'segmented_image']*2)


t_total = round((t4-t1)/60, 2)
t_load = round((t2-t1)/60, 2)
t_process = round((t3-t2)/60, 2)
t_others = round((t4-t3)/60, 2)

print()
print()
print('TOTAL TIME            : {}m.      '.format(t_total))
print('IMAGES LOAD TIME      : {}m. ({}%)'.format(t_load,    round(t_load*100/t_total, 2)))
print('IMAGES PROCESSING TIME: {}m. ({}%)'.format(t_process, round(t_process*100/t_total, 2)))
print('OTHERs TIME           : {}m. ({}%)'.format(t_others,  round(t_others*100/t_total, 2)))
print()
print()
