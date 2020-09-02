from root.utils import readImage
from root.readData import readCompleteCSV
from root.seg.chan_vese import segment
from skimage.io import imsave
import concurrent.futures
import os

input_path  = '../data/output/hr/128/'
output_path = '../data/output/seg/128/'

def segment_server():

    print()
    print()
    print('Output images will be saved into "../data/output/seg/128" folder in the following way:')
    print('image_name_seg.jpg')
    print('.')
    print('.')
    print('.')
    print()
    print()

    df = readCompleteCSV()

    images_names = df.image_name.values

    print()
    print('Number of images: {}.'.format(len(images_names)))
    print()

    for image_name in images_names:
        image = readImage(input_path + image_name + '_hr.jpg')
        out = segment(image)
        imsave(output_path + image_name + '_seg.jpg', out)
    

