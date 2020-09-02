from root.utils import readImage
from root.readData import readCompleteCSV
from root.segmentation.hr import hairs_remove
from skimage.io import imsave
import os

def hr_server():

    print()
    print()
    print('Output images will be saved into "../data/output/hr" folder in the following way:')
    print('image_name_hr.jpg')
    print('.')
    print('.')
    print('.')
    print()
    print()


    df = readCompleteCSV()

    images_names = df.image_name.values
    print(len(images_names))

    input_path  = '../data/input/jpeg/train/'
    output_path = '../data/output/hr/128/'
    
    for image_name in images_names:
    
        image = readImage(input_path + image_name + '.jpg')

        out = hairs_remove(image)

        imsave(output_path + image_name + '_hr.jpg', out)
    

