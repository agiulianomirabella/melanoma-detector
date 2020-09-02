from root.utils import digitizeToEqualWidth, readImage
from root.readData import readCompleteCSV
from root.seg.stats import stats
import pandas as pd
import os

def stats_server():

    print()
    print()
    print('Results will be written in "../data/halfway/seg/stats_128.csv" in the following way:')
    print('image_name  , stats0, stats1, ..., stats10')
    print('ISIC_0015719,    ...,    ..., ..., ...')
    print('.')
    print('.')
    print('.')
    print()
    print()

    df = readCompleteCSV()
    images_names = df.image_name.values

    out = pd.DataFrame(columns=['image_name', 'areas_ratio', 'major_axis_weighted_mean', 'minor_axis_weighted_mean', 
        'eccentricity_mean', 'eccentricity_std', 'R_mean', 'G_mean', 'B_mean', 'R_std', 'G_std', 'B_std'])

    i = 0

    for image_name in images_names:

        new_row = stats(image_name)

        new_df = pd.DataFrame.from_records([{
            'image_name':               image_name,
            'areas_ratio':              new_row[0],
            'major_axis_weighted_mean': new_row[1],
            'minor_axis_weighted_mean': new_row[2],
            'eccentricity_mean':        new_row[3],
            'eccentricity_std':         new_row[4],
            'R_mean':                   new_row[5],
            'G_mean':                   new_row[6],
            'B_mean':                   new_row[7],
            'R_std':                    new_row[8],
            'G_std':                    new_row[9],
            'B_std':                    new_row[10]
        }])

        out = out.append(new_df)
        if i%100 == 0 or i == 5 or i == 10 or i == 20 or i == 50:
            print(i)
        i = i + 1
    
    out.to_csv('../data/halfway/csv/stats_128.csv', index=False)

