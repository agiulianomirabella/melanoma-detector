from root.utils import readImage, toGray, resize
from root.readData import readCompleteCSV
from skimage.io import imsave
import os

input_path  = '../data/input/jpeg/train/'
output_path = '../data/output/gray/128/'

def gray_server():

    print()
    print()
    print('Output images will be saved into "../data/output/gray/128" folder in the following way:')
    print('image_name_gray.jpg')
    print('.')
    print('.')
    print('.')
    print()
    print()


    df = readCompleteCSV()

    images_names = df.image_name.values[:5]

    print()
    print('Number of images: {}.'.format(len(images_names)))
    print()

    for image_name in images_names:
    
        image = readImage(input_path + image_name + '.jpg')

        out = resize(toGray(image), (256, 256))

        imsave(output_path + image_name + '_gray.jpg', out)
    

