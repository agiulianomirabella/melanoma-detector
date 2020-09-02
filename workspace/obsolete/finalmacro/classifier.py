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
from root.finalmacro.plots import plot_train_hist, plot_score, write_results
from root.finalmacro.generators import multiple_inputs_generator, create_callbacks
from root.finalmacro.models import create_model

BALANCE_DATA = True
BATCH_SIZE   = 8
EPOCHS       = 5
K            = 5
METRICS = ['AUC', 'accuracy']

AUG_PARAMETERS = (
    20,
    0.15,
    0.2,
    0.2,
    0.15,
    True,
    "nearest"
)

possible_inputs = ['rgb', 'gray', 'hr', 'seg', 'categorical', 'stats', 'euler']

def melanoma_detector(model_name, save_path, n_sample, inputs= ['rgb', 'hr', 'categorical', 'euler'], regularizations= [], 
    dropouts= [], batch_size= BATCH_SIZE, epochs= EPOCHS, k = K, aug_parameters= AUG_PARAMETERS):

    if not all([a in possible_inputs for a in inputs]):
        raise Exception('Unexpected inputs, choose from {}'.format(possible_inputs))

    fold_var = 1
    all_auc_histories = []
    
    train_df, valid_df = readCSV(n_sample)
    skf = StratifiedKFold(n_splits = k, random_state = 7, shuffle = True)

    df = pd.concat([train_df, valid_df])
    df.fillna(0, inplace=True)

    for train_index, valid_index in skf.split(np.zeros(n_sample), df[['target']]):

        train_generator = multiple_inputs_generator(inputs, train_index, df, aug_parameters)
        valid_generator = multiple_inputs_generator(inputs, valid_index, df, aug_parameters)

        checkpoint = create_callbacks(model_name, save_path, fold_var)
        callbacks_list = [checkpoint]

        model = create_model(model_name, inputs= inputs, regularizations= regularizations, dropouts= dropouts)
        model.compile(optimizer= 'adam', loss= 'binary_crossentropy', metrics= METRICS)

        history = model.fit(
            x=train_generator,
            validation_data=valid_generator,
            epochs=epochs,
            callbacks=callbacks_list
        )

        all_auc_histories.append(history.history['val_AUC'])

        plot_train_hist(history, save_path, model_name + str(fold_var))
        
        #Load the best model to evaluate its performance
        '''
        model.load(save_path + model_name + str(fold_var))
        
        results = model.evaluate(valid_generator)
        results = dict(zip(model.metrics_names, results))
        
        tf.keras.backend.clear_session()
        '''        
        
        fold_var = fold_var + 1

    auc_per_epochs = [np.mean([x[i] for x in all_auc_histories]) for i in range(epochs)]
    write_results(auc_per_epochs, save_path)
