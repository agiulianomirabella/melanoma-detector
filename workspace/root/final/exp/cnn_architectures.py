import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Lambda # pylint: disable= import-error
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras import Input, Model, regularizers # pylint: disable= import-error
from sklearn.model_selection import StratifiedKFold
from root.readData import readCSV
from root.final.exp.cnn_utils import create_callbacks, get_train_valid_dataset
from root.final.exp.plots import plot_train_hist, write_experiment_parameters

METRICS     = ['AUC', 'accuracy']

def get_rgb_architecture():
    input_shape = (128, 128, 3)
    model_input = Input(shape=input_shape, name='input_layer')
    #x = Rescaling(1.0 / 255)(model_input)
    x = tf.keras.applications.EfficientNetB0(include_top=False, weights='imagenet', input_shape=input_shape, pooling='max')(model_input)
    x = Dense(16, activation='relu', kernel_regularizer= tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    
    model = Model(model_input, x, name='rgb_CNN')
    model.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= METRICS)

    #model.summary()
    
    return model

def get_gray_architecture():
    input_shape = (128, 128, 3)
    model_input = Input(shape=input_shape, name='input_layer')
    dummy = Lambda(lambda x:x)(model_input)
    
    x = tf.keras.applications.densenet.DenseNet201(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')(dummy)
    x = Dense(64, activation='relu', kernel_regularizer= tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    
    model = Model(model_input, x, name='rgb_CNN')
    model.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= METRICS)

    #model.summary()

    return model

def get_hr_architecture():
    input_shape = (128, 128, 3)
    model_input = Input(shape=input_shape, name='input_layer')
    dummy = Lambda(lambda x:x)(model_input)
    
    x = tf.keras.applications.densenet.DenseNet201(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')(dummy)
    x = Dense(64, activation='relu', kernel_regularizer= tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    
    model = Model(model_input, x, name='rgb_CNN')
    model.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= METRICS)

    #model.summary()

    return model


def get_seg_architecture():
    input_shape = (128, 128, 3)
    model_input = Input(shape=input_shape, name='input_layer')
    dummy = Lambda(lambda x:x)(model_input)
    
    x = tf.keras.applications.densenet.DenseNet201(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')(dummy)
    x = Dense(64, activation='relu', kernel_regularizer= tf.keras.regularizers.l1_l2(l1=0.001, l2=0.001))(x)
    x = Dropout(0.5)(x)
    x = Dense(1, activation='sigmoid')(x)
    
    model = Model(model_input, x, name='rgb_CNN')
    model.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= METRICS)
    
    #model.summary()

    return model


def get_model(keyword):
    if keyword == 'rgb':
        return get_rgb_architecture()
    if keyword == 'gray':
        return get_gray_architecture()
    if keyword == 'hr':
        return get_hr_architecture()
    if keyword == 'seg':
        return get_seg_architecture()

