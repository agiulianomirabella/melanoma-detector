import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint # pylint: disable= import-error
from tensorflow.keras.preprocessing.image import ImageDataGenerator # pylint: disable= import-error
from sklearn.utils.class_weight import compute_class_weight

CALLBACK_MONITOR = 'val_auc'
BATCH_SIZE  = 32
BALANCE_DATA = True

AUG_PARAMETERS = (
    20,
    0.15,
    0.2,
    0.2,
    0.15,
    True,
    "nearest"
)

def get_train_valid_dataset(keyword, df, train_index, valid_index):

    df_aux = df.copy()
    df_aux['image_name'] = df_aux.image_name.apply(lambda x: x + aux_suffix[keyword])

    rotation_range, zoom_range, width_shift_range, height_shift_range, shear_range, horizontal_flip, fill_mode = aug_parameters_aux[keyword]

    aug = ImageDataGenerator(rescale = aux_rescale[keyword], rotation_range=rotation_range, zoom_range=zoom_range,
    width_shift_range=width_shift_range, height_shift_range=height_shift_range, shear_range=shear_range,
    horizontal_flip=horizontal_flip, fill_mode=fill_mode)

    train_generator = get_generator(keyword, df_aux, train_index, aug)
    valid_generator = get_generator(keyword, df_aux, valid_index, aug)

    return train_generator, valid_generator


def get_generator(keyword, df, index, image_data_generator):


    data_generator = image_data_generator.flow_from_dataframe(
        df.iloc[index],
        aux_directory[keyword],
        x_col='image_name',
        y_col='benign_malignant',
        class_mode = 'binary',
        target_size=aux_shape[keyword],
        color_mode=aux_color_mode[keyword],
        batch_size = BATCH_SIZE,
        shuffle = True
   )

    return data_generator

def class_weights(train_df):
    y_train = train_df['target']
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {0: class_weights[0],1: class_weights[1]}
    if not BALANCE_DATA:
        class_weights = {0: 1,1: 2}
    return class_weights


def create_callbacks(model_name, save_path, fold_var):
    checkpoint = ModelCheckpoint(
        save_path + model_name + '_' +str(fold_var),
        monitor=CALLBACK_MONITOR, 
        verbose=1,
        save_best_only= True,
        save_weights_only= True,
        mode='max'
    )

    return [checkpoint]





########     AUX PARAMETERS:

RGB_INPUT_SHAPE         = (128, 128) # do not consider the color axis
GRAYSCALED_INPUT_SHAPE  = (128, 128)
HR_INPUT_SHAPE          = (128, 128)
SEG_INPUT_SHAPE         = (128, 128)


aux_shape = {
    'rgb':         RGB_INPUT_SHAPE,
    'gray':        GRAYSCALED_INPUT_SHAPE,
    'hr':          HR_INPUT_SHAPE,
    'seg':         SEG_INPUT_SHAPE
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

aug_parameters_aux = {
    'rgb':         AUG_PARAMETERS,
    'gray':        AUG_PARAMETERS,
    'hr':          AUG_PARAMETERS,
    'seg':         AUG_PARAMETERS
}
