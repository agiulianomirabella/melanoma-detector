import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator # pylint: disable= import-error
from tensorflow.keras.callbacks import ModelCheckpoint # pylint: disable= import-error
from tensorflow.keras import regularizers # pylint: disable= import-error
from root.readData import readCSV
from sklearn import metrics

'''
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
'''

'''
WARNING:tensorflow:6 out of the last 11 calls to <function Model.make_test_function.<locals>.test_function at 0x7f47cc5c5830> 
triggered tf.function retracing. Tracing is expensive and the excessive number of tracings could be due to 
(1) creating @tf.function repeatedly in a loop, 
(2) passing tensors with different shapes, 
(3) passing Python objects instead of tensors. 

For (1), please define your @tf.function outside of the loop. 
For (2), @tf.function has experimental_relax_shapes=True option that relaxes argument shapes that can avoid unnecessary retracing. 
For (3), please refer to https://www.tensorflow.org/tutorials/customization/performance#python_or_tensor_args and 
https://www.tensorflow.org/api_docs/python/tf/function for  more details.

WARNING:tensorflow:Can save best model only with val_AUC available, skipping.
'''


BALANCE_DATA = True
input_shape = (128, 128, 3)

aug_parameters = (
    20,
    0.15,
    0.2,
    0.2,
    0.15,
    True,
    "nearest"
)

def plot_train_hist(hist, path, model_name):
    # Plot training & validation accuracy values
    plt.plot(hist.history['AUC'])
    plt.plot(hist.history['val_AUC'])
    plt.title('Model accuracy')
    plt.ylabel('Auc')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(path + model_name + '_auc.jpg')
    plt.clf()

    # Plot training & validation accuracy values
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(path + model_name + '_accuracy.jpg')
    plt.clf()

    # Plot training & validation loss values
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(path + model_name + '_loss.jpg')
    plt.clf()

def create_model():
    
    model_input = tf.keras.Input(shape=input_shape, name='input_layer')
    dummy = tf.keras.layers.Lambda(lambda x:x)(model_input)
    outputs = []
    
    x = tf.keras.applications.densenet.DenseNet201(include_top=False, weights='imagenet', input_shape=input_shape, pooling='avg')(dummy)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer= regularizers.l1_l2(l1=0.001, l2=0.001))(x)    
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    outputs.append(x)
    
    model = tf.keras.Model(model_input, outputs, name='denseNet201BasedNetworkKFold')
    model.summary()
    
    return model

#tf.keras.metrics.SensitivityAtSpecificity.name,

def create_and_train_model(model_name, save_path, n_sample, batch_size, epochs, 
                                                    k, aug_parameters = aug_parameters, metrics = ['AUC', 'accuracy']):
    all_auc_histories = []
    fold_var = 1
    
    skf = StratifiedKFold(n_splits = k, random_state = 7, shuffle = True)
    train_df, valid_df = readCSV(n_sample, valid_size=0.2)
    df = pd.concat([train_df, valid_df])
    df['image_name'] = df.image_name.apply(lambda x: x+'.jpg')

    for train_index, val_index in skf.split(np.zeros(n_sample), df[['target']]):
        training_data = df.iloc[train_index]
        validation_data = df.iloc[val_index]

        # train the model on the new data for a few epochs
        aug = ImageDataGenerator(rescale = 1./255, rotation_range=20, zoom_range=0.15,
            width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
            horizontal_flip=True, fill_mode="nearest")

        train_generator = aug.flow_from_dataframe(
            training_data,
            '../data/input/jpeg/train/', 
            x_col='image_name',
            y_col='benign_malignant',
            class_mode = 'binary',
            target_size=input_shape[:2],
            color_mode='rgb',
            shuffle = True
        )

        valid_generator = aug.flow_from_dataframe(
            validation_data,
            '../data/input/jpeg/train/', 
            x_col='image_name',
            y_col='benign_malignant',
            class_mode = 'binary',
            target_size=input_shape[:2],
            color_mode='rgb',
            shuffle = True
        )

        model = create_model()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)

        #CALLBACKS
        monitor = 'val_AUC'

        checkpoint = tf.keras.callbacks.ModelCheckpoint(
                                save_path + model_name + str(fold_var),
                                monitor=monitor, verbose=1,
                                save_best_only=True, mode='max')

        callbacks_list = [checkpoint]

        #FIT THE MODEL
        history = model.fit(x=train_generator,
                    epochs=epochs,
                    callbacks=callbacks_list,
                    validation_data=valid_generator)

        all_auc_histories.append(history.history['val_AUC'])

        #PLOT HISTORY
        plot_train_hist(history, save_path, model_name + str(fold_var))
        
        #LOAD BEST MODEL to evaluate the performance of the model
        '''
        model.load_weights(save_path + model_name + str(fold_var) + ".h5")
        
        results = model.evaluate(valid_generator)
        results = dict(zip(model.metrics_names, results))
        
        tf.keras.backend.clear_session()
        '''        
        
        fold_var = fold_var + 1

    auc_per_epochs = [np.mean([x[i] for x in all_auc_histories]) for i in range(epochs)]
    f = open(save_path + 'results.txt', 'w')
    for i, auc in enumerate(auc_per_epochs):
        f.write('Epoch: {}. AUC: {}.'.format(i, auc))
    f.close()

    print(all_auc_histories)
    return auc_per_epochs

def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def plot_score(auc_per_epochs, path, model_name):
    points = smooth_curve(auc_per_epochs[10:])
    plt.plot(range(1, len(points) + 1), points)
    plt.xlabel('Epochs')
    plt.ylabel('AUC score')
    plt.savefig(path + model_name + '_auc_vs_epochs.jpg')
    plt.clf()



def print_model_parameters(model_name, save_path, n_sample, batch_size, epochs, k, aug_parameters):
    f = open(save_path + model_name + '_parameters.txt', 'w')
    f.write(model_name + '\n')
    f.write('batch_size: {}\n'.format(batch_size))
    f.write('epochs: {}\n'.format(epochs))
    f.write('k: {}\n'.format(k))
    f.write('n_sample: {}\n'.format(n_sample))

    rotation_range, zoom_range, width_shift_range, height_shift_range, shear_range, horizontal_flip, fill_mode = aug_parameters
    f.write('\ndata augmentation parameters:\n')
    f.write('         rotation_range: {}\n'.format(rotation_range))
    f.write('         zoom_range: {}\n'.format(zoom_range))
    f.write('         width_shift_range: {}\n'.format(width_shift_range))
    f.write('         height_shift_range: {}\n'.format(height_shift_range))
    f.write('         shear_range: {}\n'.format(shear_range))
    f.write('         horizontal_flip: {}\n'.format(horizontal_flip))
    f.write('         fill_mode: {}\n'.format(fill_mode))

    f.close()


