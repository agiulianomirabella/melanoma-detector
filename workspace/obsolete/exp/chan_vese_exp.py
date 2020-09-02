from root.utils import * # pylint: disable= unused-wildcard-import
from root.readData import readData
from root.seg.chan_vese import segment
import time
import random
import os

input_path = '../data/output/hr/128/'

t1 = time.time()
files = os.listdir(input_path)
names = random.sample(files, 4)
without_hairs = [readImage(input_path + img) for img in names]

print()
for i in names:
    print(i)
print()

t2 = time.time()
#without_hairs = [resize(toGray(img), (128, 128)) for img in images]

t3 = time.time()
segmented_image = [segment(img) for img in without_hairs]

t4 = time.time()

displaySeveralImages(without_hairs + segmented_image, ncols= 2,
    titles=['without_hairs', 'segmented_image']*2)

t_total   = round((t4-t1)/60, 2)
t_load    = round((t2-t1)/60, 2)
t_process = round((t3-t2)/60, 2)
t_others  = round((t4-t3)/60, 2)

print()
print()
print('TOTAL TIME            : {}m.      '.format(t_total))
print('IMAGES LOAD TIME      : {}m. ({}%)'.format(t_load,    round(t_load*100/t_total, 2)))
print('IMAGES PROCESSING TIME: {}m. ({}%)'.format(t_process, round(t_process*100/t_total, 2)))
print('OTHERs TIME           : {}m. ({}%)'.format(t_others,     round(t_others*100/t_total, 2)))
print()
print()
