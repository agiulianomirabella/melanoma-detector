from root.utils import * # pylint: disable= unused-wildcard-import
import pandas as pd
import numpy as np
import math
from skimage.color import rgb2gray
from random import sample
from sklearn.model_selection import train_test_split

def readData(n_sample, valid_size = 0.2):
    # return n instances of:
    #   - data: DataFrame
    #   - (train_images, train_labels): (numpy array, 1/0)
    #   - (valid_images , valid_labels ): (numpy array, 1/0)
    # half the instances will be benign and half malignant.

    if not isinstance(n_sample, int):
        n_sample = math.floor(n_sample)
    if n_sample%2 != 0:
        n_sample = n_sample-1
    if n_sample <= 0:
        print('WARNING: n_sample = 2 is used')
        n_sample = 2
    if n_sample > 1168:
        print('WARNING: there are no more than 584 malignant images, thus no more than 1168 images can be read to keep the target balanced')
        n_sample = 1168


    csv          = readCompleteCSV()
    df_benign    = csv[csv['target']==0].sample(int(n_sample/2))
    df_malignant = csv[csv['target']==1].sample(int(n_sample/2))

    df_train_benign,    df_valid_benign    = train_test_split(df_benign,    test_size=valid_size)
    df_train_malignant, df_valid_malignant = train_test_split(df_malignant, test_size=valid_size)

    train_df = pd.concat([df_train_benign, df_train_malignant])
    valid_df  = pd.concat([df_valid_benign,  df_valid_malignant])

    train_images = getImages(train_df.image_name.values.tolist())
    valid_images = getImages(valid_df.image_name.values.tolist())

    print()
    print('The number of train cases is:  {}. {} benign and {} malignant. '.format(
        len(train_df), len(df_train_benign), len(df_train_malignant)
    ))
    print('The number of valid  cases is: {}. {} benign and {} malignant. '.format(
        len(valid_df), len(df_valid_benign), len(df_valid_malignant)
    ))
    print()

    return train_df, train_images, train_df.target.values, valid_df, valid_images, valid_df.target.values


def readCompleteCSV():
    df = pd.read_csv('../data/halfway/csv/reduced_train.csv')
    return df

def readCSV(n_sample, test_size= None, shuffle = True):
    if n_sample > 1168:
        print()
        print()
        print('WARNING: there are no more than 584 malignant images, thus no more than 1168 cases can be loaded to keep the target balance')
        print()
        print()
        n_sample = 1168
    csv = pd.read_csv('../data/halfway/csv/submodels.csv')
    df_benign    = csv[csv['target']==0].sample(int(n_sample/2))
    df_malignant = csv[csv['target']==1].sample(int(n_sample/2))

    if test_size:
        df_train_benign,    df_valid_benign    = train_test_split(df_benign,    test_size=test_size)
        df_train_malignant, df_valid_malignant = train_test_split(df_malignant, test_size=test_size)

        train_df = pd.concat([df_train_benign, df_train_malignant])
        valid_df = pd.concat([df_valid_benign, df_valid_malignant])
        
        if shuffle:
            train_df = train_df.sample(frac = 1)
            valid_df = valid_df.sample(frac = 1)
        return train_df, valid_df

    else:
        train_df = pd.concat([df_benign, df_malignant])
        if shuffle:
            train_df = train_df.sample(frac = 1)
        return train_df



def readCSV_imbalanced(n_sample, malignant_test_size= 0.15, shuffle = True):
    if n_sample > 5509:
        n_sample = 5509
    if n_sample < 1168:
        n_sample = 1168

    csv = pd.read_csv('../data/halfway/csv/submodels.csv')

    df_benign    = csv[csv['target']==0].sample(n_sample - 584)
    df_malignant = csv[csv['target']==1]

    df_train_malignant, df_test_malignant = train_test_split(df_malignant, test_size= malignant_test_size)
    df_train_benign,    df_test_benign    = train_test_split(df_benign,    test_size= len(df_test_malignant.index)/len(df_benign.index))

    train_df = pd.concat([df_train_benign, df_train_malignant])
    test_df = pd.concat([df_test_benign, df_test_malignant])
    
    if shuffle:
        train_df = train_df.sample(frac = 1)
        test_df = test_df.sample(frac = 1)
    return train_df, test_df
