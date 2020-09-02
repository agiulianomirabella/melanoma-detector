import pandas as pd
import numpy as np
import os
import random

final_csv_path = '../data/halfway/csv/'

def create_final_csv():

    df    = pd.read_csv('../data/halfway/csv/reduced_train.csv')
    stats = pd.read_csv('../data/halfway/csv/stats_128.csv')
    euler = pd.read_csv('../data/halfway/csv/euler_128.csv')

    df.sort_values('image_name', inplace=True, ignore_index = True)
    stats.sort_values('image_name', inplace=True, ignore_index = True)
    euler.sort_values('image_name', inplace=True, ignore_index = True)

    df.drop(['patient_id', 'diagnosis'], axis=1)

    final = df.merge(stats, on = 'image_name', validate = 'one_to_one')
    final = final.merge(euler, on = 'image_name', validate = 'one_to_one')

    columns_titles = [
        'image_name', 'sex', 'anatom_site_general_challenge', 'age_approx',
        'areas_ratio', 'major_axis_weighted_mean', 'minor_axis_weighted_mean', 'eccentricity_mean', 'eccentricity_std',
        'R_mean', 'G_mean', 'B_mean', 'R_std', 'G_std', 'B_std',
        'euler0', 'euler1', 'euler2', 'euler3', 'euler4',
        'benign_malignant', 'target'
    ]

    final=final.reindex(columns=columns_titles)

    final.fillna(0, inplace=True)

    print(final.head())

    final.to_csv(final_csv_path + 'submodels.csv', index = False)

    csv_checkup()

def csv_checkup():
    df    = pd.read_csv('../data/halfway/csv/reduced_train.csv')
    stats = pd.read_csv('../data/halfway/csv/stats_128.csv')
    euler = pd.read_csv('../data/halfway/csv/euler_128.csv')
    final = pd.read_csv(final_csv_path + 'submodels.csv')

    df.fillna(0, inplace=True)
    stats.fillna(0, inplace=True)
    euler.fillna(0, inplace=True)
    final.fillna(0, inplace=True)

    df.sort_values('image_name'   , inplace=True, ignore_index = True)
    stats.sort_values('image_name', inplace=True, ignore_index = True)
    euler.sort_values('image_name', inplace=True, ignore_index = True)
    final.sort_values('image_name', inplace=True, ignore_index = True)

    df.drop(['patient_id', 'diagnosis'], axis=1)
    columns_titles = [
        'image_name', 'sex', 'anatom_site_general_challenge', 'age_approx', 'benign_malignant', 'target'
    ]
    df=df.reindex(columns=columns_titles)

    for i in df.image_name.values:
        if not all(final[final['image_name']==i].iloc[:, :4] == df[df['image_name']==i].iloc[:, :4]):
            raise Exception('ERROR in image_name: {}'.format(i))
        if not all(final[final['image_name']==i].iloc[:, 20:] == df[df['image_name']==i].iloc[:, 4:]):
            raise Exception('ERROR in image_name: {}'.format(i))
        if not all(final[final['image_name']==i].iloc[:, 4:15] == stats[stats['image_name']==i].iloc[:, 1:]):
            raise Exception('ERROR in image_name: {}'.format(i))
        if not all(final[final['image_name']==i].iloc[:, 15:20] == euler[euler['image_name']==i].iloc[:, 1:]):
            raise Exception('ERROR in image_name: {}'.format(i))

    print('EVERITHING IS OK')

if __name__ == "__main__":
    create_final_csv()

