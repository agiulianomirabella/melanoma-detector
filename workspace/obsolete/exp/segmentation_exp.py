from root.utils import * # pylint: disable= unused-wildcard-import
from root.readData import readData
from root.seg.chan_vese_display import segment
from root.seg.hr import hairs_remove
import time
import os
import random

t1 = time.time()
input_path = '../data/input/jpeg/train/'
img_names = random.sample(os.listdir(input_path), 2)
print()
for i in img_names:
    print(i)
print()
images = [resize(readImage(input_path + img_name), (512, 512)) for img_name in img_names]

without_hairs = [hairs_remove(img) for img in images]

t2 = time.time()
outputs = [segment(img, extended_output=True) for img in without_hairs]
t3 = time.time()

grayScale     = [e[0] for e in outputs]
grayScale_aux = [e[1] for e in outputs]
out           = [e[2] for e in outputs]
out_aux       = [e[3] for e in outputs]

first   = [images[0], grayScale[0], grayScale_aux[0], out[0], out_aux[0]]
second  = [images[1], grayScale[1], grayScale_aux[1], out[1], out_aux[1]]

t4 = time.time()

displaySeveralImages(first + second, ncols= 5, 
    titles=['original', 'hairs removed', 'circle mask', 'segmented', 'opened']*2)


print()
print()
print('TIEMPO TOTAL:                {}s.'.format(round(t4-t1, 2)))
print('TIEMPO DE SEGMENTACIÓN:      {}s.'.format(round(t3-t2, 2)))
print('PORCENTAJE:                  {}%.'.format(round((t3-t2)*100/(t4-t1), 2)))
print()
print()
