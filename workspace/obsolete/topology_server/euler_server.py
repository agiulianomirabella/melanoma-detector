from root.utils import digitizeToEqualWidth, readImage
from root.topology.euler import eulerInfo
import os
import pandas as pd

def euler_server():

    print()
    print()
    print('Results will be written in "../data/output/euler/euler_hr_128.csv" in the following way:')
    print('image_name, euler0, euler1, euler2, euler3, euler4')
    print('ISIC_0015719, ..., ..., ..., ..., ...')
    print('.')
    print('.')
    print('.')
    print()
    print()

    input_path = '../data/output/hr/128/'
    suffix = '_hr.jpg'

    out = pd.DataFrame(columns=['image_name', 'euler0', 'euler1', 'euler2', 'euler3', 'euler4'])

    for file_name in os.listdir(input_path)[:100]:

        image_name = file_name[:-len(suffix)]

        image = readImage(input_path + file_name)

        euler_info = eulerInfo(image)

        new_df = pd.DataFrame.from_records([{
            'image_name': image_name,
            'euler0': euler_info[0],
            'euler1': euler_info[1],
            'euler2': euler_info[2],
            'euler3': euler_info[3],
            'euler4': euler_info[4]
            }])

        out = out.append(new_df)
    
    out.to_csv('../data/output/euler/euler_hr_128.csv', index=False)

