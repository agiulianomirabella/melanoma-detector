
'''
../data/
    input/
        jpeg/
            train/
                ...jpg
            test/
                ...jpg
        test/
            ...dcm
        train/
            ..dcm

        complete_train.csv

    output/
        hr/
            128/
                ...jpg
            256/
                ...jpg
        euler/
            euler_hr_128.csv
            euler_hr_256.csv
        segmented/
            128/
                ...jpg
            256/
                ...jpg

        train.csv

'''

import pandas as pd
import os

cases_number = 5509

path = '../data/output/'
path_hr      = path + 'hr/'
path_hr_128  = path + 'hr/128/'
path_hr_256  = path + 'hr/256/'
path_seg     = path + 'segmented/'
path_seg_128 = path + 'segmented/128/'
path_seg_256 = path + 'segmented/256/'
path_eu      = path + 'euler/'

all_paths_lists = [
    path,
    path_hr,
    path_hr_128,
    path_hr_256,
    path_seg,
    path_seg_128,
    path_seg_256,
    path_eu
]

files         = os.listdir(path)
files_hr      = os.listdir(path_hr)
files_hr_128  = os.listdir(path_hr_128)
files_hr_256  = os.listdir(path_hr_256)
files_seg     = os.listdir(path_seg)
files_seg_128 = os.listdir(path_seg_128)
files_seg_256 = os.listdir(path_seg_256)
files_eu      = os.listdir(path_eu)

all_files_lists = [
    files,
    files_hr,
    files_hr_128,
    files_hr_256,
    files_seg,
    files_seg_128,
    files_seg_256, 
    files_eu
]

expected_files_number = {
    path: 4,
    path_hr: 2,
    path_hr_128: cases_number,
    path_hr_256: cases_number,
    path_seg: 2,
    path_seg_128: cases_number,
    path_seg_256: cases_number,
    path_eu: 2
}

contains_images = [files_hr_128, files_hr_256, files_seg_128, files_seg_256]
contains_images_paths = [path_hr_128, path_hr_256, path_seg_128, path_seg_256]

for i, p in all_paths_lists:
    if len(all_files_lists[i]) != expected_files_number[p]:
        raise Exception('ERROR: files missing from {} folder.'.format(p))

df = pd.read_csv(path + 'train.csv')
benign = df[df['target']==0]
malignant = df[df['target']==1]
images_names = list(df.image_name.values)

if cases_number != len(df.index):
    raise Exception('ERROR: the dataframe does not contain {} cases as expected.'.format(cases_number))

for image_name in images_names:
    for i, f in contains_images:
        if image_name not in f:
            raise Exception('ERROR: {} image is missing from {} folder.'.format(image_name, contains_images_paths[i]))

print()
print()
print('Total number of cases: {}'.format(len(df.index)))
print('BENIGN               :  {}'.format(len(benign.index)))
print('MALIGNANT            :   {}'.format(len(malignant.index)))
print('EVERYTHING IS OK, NOW KEEP WORKING...')
print()
print()



