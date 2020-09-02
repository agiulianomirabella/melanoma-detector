import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics

from tensorflow.keras.preprocessing.image import ImageDataGenerator # pylint: disable= import-error
from tensorflow.keras.callbacks import ModelCheckpoint # pylint: disable= import-error
from tensorflow.keras import regularizers # pylint: disable= import-error
from tensorflow.keras.layers import Dense, Dropout, Lambda, Concatenate, BatchNormalization # pylint: disable= import-error
from tensorflow.keras import Input, Model # pylint: disable= import-error
from tensorflow.keras.applications.densenet import DenseNet201 # pylint: disable= import-error

from root.readData import readCSV
from root.final.plots import plot_train_hist, plot_score

CALLBACK_MONITOR = 'val_AUC'
BATCH_SIZE = 12

RGB_INPUT_SHAPE         = (128, 128) # do not consider the color axis
GRAYSCALED_INPUT_SHAPE  = (128, 128)
HR_INPUT_SHAPE          = (128, 128)
SEG_INPUT_SHAPE         = (128, 128)
CATEGORICAL_INPUT_SHAPE = (10,)
STATS_INPUT_SHAPE       = (10,)
EULER_INPUT_SHAPE       = (5,)

AUG_PARAMETERS = (
    20,
    0.15,
    0.2,
    0.2,
    0.15,
    True,
    "nearest"
)

aux_shape = {
    'rgb':         RGB_INPUT_SHAPE,
    'gray':        GRAYSCALED_INPUT_SHAPE,
    'hr':          HR_INPUT_SHAPE,
    'seg':         SEG_INPUT_SHAPE,
    'categorical': CATEGORICAL_INPUT_SHAPE,
    'stats':       STATS_INPUT_SHAPE,
    'euler':       EULER_INPUT_SHAPE
}

aux_color_mode = {
    'rgb':         'rgb',
    'gray':        'rgb',
    'hr':          'rgb',
    'seg':         'rgb'
}

aux_suffix = {
    'rgb':         '.jpg',
    'gray':        '_gray.jpg',
    'hr':          '_hr.jpg',
    'seg':         '_seg.jpg'
}

aux_rescale = {
    'rgb':         255.,
    'gray':        255.,
    'hr':          255.,
    'seg':         1.
}

aux_directory = {
    'rgb':         '../data/input/jpeg/train/',
    'gray':        '../data/halfway/gray/128/',
    'hr':          '../data/halfway/hr/128/',
    'seg':         '../data/halfway/seg/128/'
}

aux_columns = {
    'categorical': slice(5),
    'stats':       slice(8, 19),
    'euler':       slice(19, None)
}

numerical_inputs = ['categorical', 'stats', 'euler']
images_inputs    = ['rgb', 'gray', 'hr', 'seg']


def multiple_inputs_generator(inputs, indices, df, aug_parameters= AUG_PARAMETERS):

    if len(inputs) == 1:
        if inputs[0] in images_inputs:
            return images_generator(inputs[0], indices, df, aug_parameters = aug_parameters)
        else:
            return rows_generator(inputs[0], indices, df)

    flag_first_image_input = False

    generators = []
    for keyword in inputs:
        if keyword in images_inputs:
            generators.append(images_generator(keyword, indices, df, aug_parameters))
            if not flag_first_image_input:
                first_image_input_index = len(generators) - 1
                flag_first_image_input = True
        else:
            generators.append(rows_generator(keyword, indices, df))

    while True:
        instances = [gen.__next__() for gen in generators]
        yield [inst[0] for inst in instances], instances[first_image_input_index][1]



def rows_generator(keyword, indices, df):

    data   = df.iloc[indices, aux_columns[keyword]]
    return data.iterrows()[2]


def images_generator(keyword, indices, df, aug_parameters= AUG_PARAMETERS):

    df_aux = df.copy()
    df_aux['image_name'] = df.image_name.apply(lambda x: x + aux_suffix[keyword])

    data   = df_aux.iloc[indices]
    
    # train the model on the new data for a few epochs
    rotation_range, zoom_range, width_shift_range, height_shift_range, shear_range, horizontal_flip, fill_mode = aug_parameters
    
    aug = ImageDataGenerator(rescale = 1./aux_rescale[keyword], rotation_range=rotation_range, zoom_range=zoom_range,
        width_shift_range=width_shift_range, height_shift_range=height_shift_range, shear_range=shear_range,
        horizontal_flip=horizontal_flip, fill_mode=fill_mode)

    result_generator = aug.flow_from_dataframe(
        data,
        aux_directory[keyword],
        x_col       = 'image_name',
        y_col       = 'benign_malignant',
        class_mode  = 'binary',
        batch_size  = BATCH_SIZE,
        target_size = aux_shape[keyword],
        color_mode  = aux_color_mode[keyword],
        shuffle     = True
    )
    
    return result_generator



def create_callbacks(model_name, save_path, fold_var):
    checkpoint = ModelCheckpoint(
        save_path + model_name + str(fold_var),
        monitor=CALLBACK_MONITOR, verbose=1,
        save_best_only=True, mode='max'
    )

    return checkpoint


