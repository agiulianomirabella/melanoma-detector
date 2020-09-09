from root.readData import readCSV_imbalanced, readCSV
from root.final.exp.mlp_utils import create_callbacks, get_train_valid_dataset, get_class_weights
from root.final.exp.plots import plot_train_hist, plot_experiment_means, write_experiment_parameters
from root.final.exp.mlp_architectures import get_model

import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential # pylint: disable= import-error
from tensorflow.keras.layers import Dense, Dropout # pylint: disable= import-error
from tensorflow.keras import Input, Model # pylint: disable= import-error
from sklearn.model_selection import StratifiedKFold
from keras.utils.vis_utils import plot_model
from keras.metrics import AUC, Accuracy # pylint: disable= import-error


BATCH_SIZE  = 32
EPOCHS      = 50
K           = 5
N_SAMPLE    = 1168
METRICS     = ['AUC', 'accuracy']


def create_mlp_experiment(model_name, keyword, n_sample= N_SAMPLE, batch_size= BATCH_SIZE, epochs= EPOCHS, k= K):

    SAVE_PATH = '../data/exp/submodels/'
    SAVE_PATH = SAVE_PATH + keyword + '/'

    if not os.path.exists(SAVE_PATH + keyword + '_plots'):
        os.mkdir(SAVE_PATH + keyword + '_plots')

    save_name = keyword + '_' + model_name

    if os.path.exists(SAVE_PATH + save_name):
        shutil.rmtree(SAVE_PATH + save_name)
    os.mkdir(SAVE_PATH + save_name)
    save_path = SAVE_PATH + save_name + '/'
    os.mkdir(save_path + 'K_plots/')

    write_experiment_parameters(save_name, save_path, n_sample, batch_size, epochs, k)

    #df, test_df = readCSV_imbalanced(n_sample)
    df = readCSV(n_sample)
    skf = StratifiedKFold(n_splits = k, shuffle = True)
    folder = 0

    class_weights = get_class_weights(df)

    all_auc_histories = []
    all_acc_histories = []
    all_loss_histories = []
    all_val_auc_histories = []
    all_val_acc_histories = []
    all_val_loss_histories = []

    for train_index, valid_index in skf.split(np.zeros(len(df.index)), df[['target']]):

        print()
        print()
        print('Model name: {}. k experiment: {}/{}'.format(save_name, folder+1, k))
        print()
        print()

        train_data, valid_data = get_train_valid_dataset(keyword, df, train_index, valid_index, batch_size)
        model = get_model(keyword, train_data)

        tf.keras.utils.plot_model(model, to_file=save_path + 'model_architecture.png', show_shapes=True, rankdir="LR")

        history = model.fit(
            x = train_data,
            validation_data = valid_data,
            batch_size = batch_size,
            epochs = epochs
        )

        plot_train_hist(history, save_path + 'K_plots/', save_name + '_K' + str(folder))

        all_auc_histories.append(history.history['auc'])
        all_acc_histories.append(history.history['accuracy'])
        all_loss_histories.append(history.history['loss'])
        all_val_auc_histories.append(history.history['val_auc'])
        all_val_acc_histories.append(history.history['val_accuracy'])
        all_val_loss_histories.append(history.history['val_loss'])

        folder = folder + 1

    mean_auc_per_epoch = [np.mean([x[i] for x in all_auc_histories]) for i in range(epochs)]
    mean_acc_per_epoch = [np.mean([x[i] for x in all_acc_histories]) for i in range(epochs)]
    mean_loss_per_epoch = [np.mean([x[i] for x in all_loss_histories]) for i in range(epochs)]
    mean_val_auc_per_epoch = [np.mean([x[i] for x in all_val_auc_histories]) for i in range(epochs)]
    mean_val_acc_per_epoch = [np.mean([x[i] for x in all_val_acc_histories]) for i in range(epochs)]
    mean_val_loss_per_epoch = [np.mean([x[i] for x in all_val_loss_histories]) for i in range(epochs)]

    plot_experiment_means(
        mean_auc_per_epoch, mean_acc_per_epoch, mean_loss_per_epoch,
        mean_val_auc_per_epoch, mean_val_acc_per_epoch, mean_val_loss_per_epoch, 
        SAVE_PATH + keyword + '_plots/', save_name,
        batch_size, keyword, model_name
    )


