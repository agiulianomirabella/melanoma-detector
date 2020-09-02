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
from tensorflow.keras.layers import Dense, Dropout, Lambda, concatenate, BatchNormalization # pylint: disable= import-error
from tensorflow.keras import Input, Model, Sequential # pylint: disable= import-error
from tensorflow.keras.applications.densenet import DenseNet201, DenseNet169, DenseNet121# pylint: disable= import-error

from root.readData import readCSV
from root.final.plots import plot_train_hist, plot_score

L1_DEFECT      = 0.001
L2_DEFECT      = 0.001
DROPOUT_DEFECT = 0.5
AUG_PARAMETERS = (
    20,
    0.15,
    0.2,
    0.2,
    0.15,
    True,
    "nearest"
)

RGB_INPUT_SHAPE         = (128, 128, 3)
GRAYSCALED_INPUT_SHAPE  = (128, 128, 3)
HR_INPUT_SHAPE          = (128, 128)
SEG_INPUT_SHAPE         = (128, 128)
CATEGORICAL_INPUT_SHAPE = (10,)
STATS_INPUT_SHAPE       = (10,)
EULER_INPUT_SHAPE       = (5,)

aux_shape = {
    'rgb':         RGB_INPUT_SHAPE,
    'gray':        GRAYSCALED_INPUT_SHAPE,
    'hr':          HR_INPUT_SHAPE,
    'seg':         SEG_INPUT_SHAPE,
    'categorical': CATEGORICAL_INPUT_SHAPE,
    'stats':       STATS_INPUT_SHAPE,
    'euler':       EULER_INPUT_SHAPE
}

images_inputs    = ['rgb', 'gray', 'hr', 'seg']

def create_cnn(keyword, regularization, dropout):
    input_shape = aux_shape[keyword]
    #outputs = []

    input_layer = Input(shape=input_shape)
    dummy       = Lambda(lambda x:x)(input_layer)

    if keyword == 'rgb':
        x = DenseNet201(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')(dummy)
    elif keyword == 'hr':
        x = DenseNet121(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')(dummy)
    elif keyword == 'gray':
        x = DenseNet169(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')(dummy)

    
    if regularization:
        x = Dense(64, activation='relu',
            kernel_regularizer= regularizers.l1_l2(l1=regularization[0], l2=regularization[1]))(x)
    else:
        x = Dense(64, activation='relu')(x)

    x = Dense(32, activation='relu')(x)

    if not dropout is None:
        x = Dropout(dropout)(x)

    x = Dense(1, activation='sigmoid')(x)
    #outputs.append(x)

    return Model(inputs= input_layer, outputs= x, name= keyword + '_densenet201')


def create_mlp(keyword, regularization, dropout):
    input_shape = aux_shape[keyword]

    input_layer = Input(shape=input_shape)

    x = Dense(64, activation='relu')(input_layer)
    
    if regularization:
        x = Dense(32, activation='relu',
            kernel_regularizer= regularizers.l1_l2(l1=regularization[0], l2=regularization[1]))(x)
    else:
        x = Dense(32, activation='relu')(x)

    if not dropout is None:
        x = Dropout(dropout)(x)

    x = Dense(1, activation='sigmoid')(x)

    return Model(inputs= input_layer, outputs=x)


def create_model(model_name, inputs, regularizations, dropouts):

    for keyword in inputs:
        if not keyword in aux_shape.keys():
            raise Exception('Unexpected input {}. Choose from: {}'.format(keyword, aux_shape.keys()))

    if regularizations and len(inputs) != len(regularizations):
        raise Exception('inputs length is different from regularizations length')
    if dropouts and len(inputs) != len(dropouts):
        raise Exception('inputs length is different from dropouts length')
    if not all([isinstance(e, tuple) for e in regularizations]):
        raise Exception('regularizations elements must be tuples of (l1, l2) for each submodel or None')

    submodels = []

    if not regularizations:
        regularizations = [None]*len(inputs)

    if not dropouts:
        dropouts        = [None]*len(inputs)

    if len(inputs) == 1:
        final_model = Sequential()
        if inputs[0] in images_inputs:
            final_model.add(create_cnn(inputs[0], regularizations[0], dropouts[0]))
        else:
            final_model.add(create_mlp(inputs[0], regularizations[0], dropouts[0]))
    else:
        for i, input_name in enumerate(inputs):
            if input_name in images_inputs:
                submodels.append(create_cnn(input_name, regularizations[i], dropouts[i]))
            else:
                submodels.append(create_mlp(input_name, regularizations[i], dropouts[i]))

    if len(inputs) > 1:
        combined_input = concatenate([m.output for m in submodels])
        last_layers = Dense(len(inputs), activation="relu")(combined_input)
        last_layers = Dense(1          , activation="sigmoid")(last_layers)
        final_model = Model(inputs = [m.input for m in submodels], outputs = last_layers)

    final_model.summary()
    
    return final_model
