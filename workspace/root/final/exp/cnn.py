import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold

from root.readData import readCSV
from root.final.exp.cnn_utils import create_callbacks, get_train_valid_dataset
from root.final.exp.cnn_architectures import get_model, create_callbacks
from root.final.exp.plots import plot_train_hist, write_experiment_parameters, plot_experiment_means

BALANCE_DATA = True
BATCH_SIZE  = 32
EPOCHS      = 15
K           = 5
N_SAMPLE    = 1168
METRICS     = ['AUC', 'accuracy']

def create_cnn_experiment(model_name, keyword, n_sample= N_SAMPLE, batch_size= BATCH_SIZE, epochs= EPOCHS, k= K):

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

    df = readCSV(n_sample)
    skf = StratifiedKFold(n_splits = K, shuffle = True)
    folder = 0

    all_auc_histories = []
    all_acc_histories = []
    all_loss_histories = []
    all_val_auc_histories = []
    all_val_acc_histories = []
    all_val_loss_histories = []

    for train_index, valid_index in skf.split(np.zeros(n_sample), df[['target']]):

        print()
        print()
        print('Model name: {}. k experiment: {}/{}'.format(save_name, folder+1, k))
        print()
        print()

        train_generator, valid_generator = get_train_valid_dataset(keyword, df, train_index, valid_index)
        model = get_model(keyword)

        history = model.fit(
            x = train_generator,
            validation_data = valid_generator,
            epochs = epochs,
            callbacks = create_callbacks(model_name, save_path, folder)
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


