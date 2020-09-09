import os
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
import skimage.io as io

from root.readData import readCSV
from root.utils import resize, normalize
from root.final.detector.cnn_utils import create_callbacks
from root.final.detector.mlp_utils import get_train_valid_dataset
from root.final.exp.cnn_architectures import create_callbacks
from root.final.exp.cnn_architectures import get_model as get_CNN
from root.final.exp.mlp_architectures import get_model as get_MLP
from root.final.exp.plots import plot_train_hist, write_experiment_parameters, plot_experiment_means

from tensorflow.keras.preprocessing.image import ImageDataGenerator # pylint: disable= import-error

BALANCE_DATA = True
BATCH_SIZE_CNN  = 16
EPOCHS_CNN      = 50

BATCH_SIZE_MLP  = 32
EPOCHS_MLP      = 50

N_SAMPLE    = 1168
METRICS     = ['AUC', 'accuracy']

keyword_MLP = 'all'
keyword_CNN = 'rgb'
CSV_PATH = '../data/output/csv/'
SAVE_PATH = '../data/output/exp/'

AUG_PARAMETERS = (
    20,
    0.15,
    0.2,
    0.2,
    0.15,
    True,
    "nearest"
)

def final_detector_cnn(model_name, csv_number, n_sample= N_SAMPLE, batch_size= BATCH_SIZE_CNN, epochs= EPOCHS_CNN):

    keyword = keyword_CNN + '_' + keyword_MLP

    if os.path.exists(SAVE_PATH + keyword + '_' + str(csv_number)):
        shutil.rmtree(SAVE_PATH + keyword + '_' + str(csv_number))
    os.mkdir(SAVE_PATH + keyword + '_' + str(csv_number))

    save_path = SAVE_PATH + keyword + '_' + str(csv_number)
    save_name = model_name + '_' + str(csv_number)

    write_experiment_parameters(save_name, save_path, n_sample, batch_size, epochs, k=-1)

    df = pd.read_csv(CSV_PATH + 'train' + str(csv_number) + '.csv')
    test_df = pd.read_csv(CSV_PATH + 'test' + str(csv_number) + '.csv')

    print()
    print()
    print('Model name: {}'.format(save_name))
    print()
    print()

    #CNN:

    df_aux = df.copy()
    df_aux['image_name'] = df_aux.image_name.apply(lambda x: x + '.jpg')

    rotation_range, zoom_range, width_shift_range, height_shift_range, shear_range, horizontal_flip, fill_mode = AUG_PARAMETERS

    aug = ImageDataGenerator(rescale = 1.0/255, rotation_range=rotation_range, zoom_range=zoom_range,
    width_shift_range=width_shift_range, height_shift_range=height_shift_range, shear_range=shear_range,
    horizontal_flip=horizontal_flip, fill_mode=fill_mode)

    train_generator = aug.flow_from_dataframe(
        df_aux,
        '../data/input/jpeg/train/',
        x_col='image_name',
        y_col='benign_malignant',
        class_mode = 'binary',
        target_size=(128, 128),
        color_mode='rgb',
        batch_size = batch_size,
        shuffle = True
   )


    model_CNN = get_CNN(keyword_CNN)
    tf.keras.utils.plot_model(model_CNN, to_file=save_path + 'model_architecture.png', show_shapes=True, rankdir="LR")

    model_CNN.fit(
        x = train_generator,
        epochs = epochs
    )

    #PREDICT

    train_images_names = df.image_name.values
    test_images_names = test_df.image_name.values

    train_predictions = []
    test_predictions = []

    for name in test_images_names:
        img = normalize(resize(io.imread('../data/input/jpeg/train/' + name + '.jpg'), (128, 128)))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        test_predictions.append(model_CNN.predict(img_array)[0][0])

    for name in train_images_names:
        img = normalize(resize(io.imread('../data/input/jpeg/train/' + name + '.jpg'), (128, 128)))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        train_predictions.append(model_CNN.predict(img_array)[0][0])


    #ADD TO CSV

    df['cnn_prediction'] = train_predictions
    test_df['cnn_prediction'] = test_predictions

    df.to_csv('../data/output/csv/train' + str(csv_number) + 'extended.csv', index=False)
    test_df.to_csv('../data/output/csv/test' + str(csv_number) + 'extended.csv', index=False)



def final_detector_mlp(model_name, csv_number, batch_size= BATCH_SIZE_MLP, epochs= EPOCHS_MLP):

    keyword = keyword_CNN + '_' + keyword_MLP

    if not os.path.exists(SAVE_PATH + keyword + '_plots'):
        os.mkdir(SAVE_PATH + keyword + '_plots')

    save_name = keyword + '_' + model_name
    if os.path.exists(SAVE_PATH + save_name):
        shutil.rmtree(SAVE_PATH + save_name)
    os.mkdir(SAVE_PATH + save_name)
    save_path = SAVE_PATH + save_name + '/'
    os.mkdir(save_path + 'plots/')

    write_experiment_parameters(save_name, save_path, -1, batch_size, epochs, k=-1)

    df = pd.read_csv(CSV_PATH + 'train' + str(csv_number) + 'extended.csv')
    test_df = pd.read_csv(CSV_PATH + 'test' + str(csv_number) + 'extended.csv')

    print()
    print()
    print('Model name: {}'.format(save_name))
    print()
    print()

    train_data, test_data = get_train_valid_dataset(keyword_MLP, df, test_df, batch_size)
    model = get_MLP(keyword_MLP, train_data)

    tf.keras.utils.plot_model(model, to_file=save_path + 'model_architecture.png', show_shapes=True, rankdir="LR")

    history = model.fit(
        x = train_data,
        validation_data= test_data,
        batch_size = batch_size,
        epochs = epochs
    )

    plot_train_hist(history, save_path + 'plots/', save_name)

    plot_experiment_means(
        history.history['auc'], history.history['accuracy'], history.history['loss'],
        history.history['val_auc'], history.history['val_accuracy'], history.history['val_loss'],
        SAVE_PATH + keyword + '_plots/', save_name,
        batch_size, keyword, model_name
    )



