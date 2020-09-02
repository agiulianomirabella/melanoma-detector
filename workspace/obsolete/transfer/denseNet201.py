import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator # pylint: disable= import-error

EPOCHS = 15
BATCH_SIZE = 32
BALANCE_DATA = True
IN_SERVER = True
input_shape = (128, 128, 3)

input_path = 'input-data/'

'''
def class_weights(train_df):
    y_train = train_df['target']
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {0: class_weights[0],1: class_weights[1]}
    if not BALANCE_DATA:
        class_weights = {0: 1,1: 2}
    return class_weights

def plot_roc(y_true, y_score):
    """
    """
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true.ravel(), y_score.ravel()) # pylint: disable= undefined-variable
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"]) # pylint: disable= undefined-variable
    
    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2]) # pylint: disable= undefined-variable
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
'''

def plot_train_hist(hist, path, model_name):
    # Plot training & validation accuracy values
    if IN_SERVER:
        plt.plot(hist.history['auc'])
        plt.plot(hist.history['val_auc'])
    else:
        plt.plot(hist.history['AUC'])
        plt.plot(hist.history['val_AUC'])
    plt.title('Model accuracy')
    plt.ylabel('Auc')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(path + model_name + '_auc.jpg')
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
    x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    outputs.append(x)
    
    model = tf.keras.Model(model_input, outputs, name='denseNet201BasedNetwork')
    model.summary()
    
    return model

aug_parameters = (
    20,
    0.15,
    0.2,
    0.2,
    0.15,
    True,
    "nearest"
)

def fit_model(model, data, epochs = EPOCHS, batch_size = BATCH_SIZE, metrics = ['AUC'], aug_parameters = aug_parameters):

    train_df, valid_df = data
    train_df['image_name'] = train_df.image_name.apply(lambda x: x+'.jpg')
    valid_df['image_name'] = valid_df.image_name.apply(lambda x: x+'.jpg')

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])

    # train the model on the new data for a few epochs
    aug = ImageDataGenerator(rescale = 1./255, rotation_range=20, zoom_range=0.15,
        width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
        horizontal_flip=True, fill_mode="nearest")

    train_generator = aug.flow_from_dataframe(
        train_df,
        '../data/input/jpeg/train/', 
        x_col='image_name',
        y_col='benign_malignant',
        class_mode = 'binary',
        target_size=input_shape[:2],
        color_mode='rgb'
        )

    valid_generator = aug.flow_from_dataframe(
        valid_df,
        '../data/input/jpeg/train/', 
        x_col='image_name',
        y_col='benign_malignant',
        class_mode = 'binary',
        target_size=input_shape[:2],
        color_mode='rgb'
        )

    # train the network
    history = model.fit(x=train_generator,
        validation_data=valid_generator,
        steps_per_epoch=train_generator.n//batch_size,
        epochs=epochs)

    return history

