from root.utils import readImage
from root.readData import readCompleteCSV
from root.segmentation.hr_256_rgb import hairs_remove_256_rgb
from skimage.io import imsave
import concurrent.futures
import os

input_path  = '../data/input/jpeg/train/'
output_path = '../data/output/hr/256-rgb/'

def aux(image_name):
    image = readImage(input_path + image_name + '.jpg')
    out = hairs_remove_256_rgb(image)
    imsave(output_path + image_name[:-4] + '_hr_256_rgb.jpg', out)


def hr_server_256_rgb():

    print()
    print()
    print('Output images will be saved into "../data/output/hr/256-rgb" folder in the following way:')
    print('image_name_hr_256_rgb.jpg')
    print('.')
    print('.')
    print('.')
    print()
    print()

    df = readCompleteCSV()

    images_names = df.image_name.values[:100]

    print()
    print('Number of images: {}.'.format(len(images_names)))
    print()

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(aux, images_names)
    

