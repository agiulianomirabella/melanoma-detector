import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Input # pylint: disable= import-error
from tensorflow.keras.callbacks import ModelCheckpoint # pylint: disable= import-error
from tensorflow.keras.layers import concatenate # pylint: disable= import-error
from tensorflow.keras.layers.experimental.preprocessing import Normalization # pylint: disable= import-error
from tensorflow.keras.layers.experimental.preprocessing import CategoryEncoding # pylint: disable= import-error
from tensorflow.keras.layers.experimental.preprocessing import StringLookup # pylint: disable= import-error
from sklearn.utils.class_weight import compute_class_weight

CALLBACK_MONITOR = 'val_auc'
BALANCE_DATA = False

def get_train_valid_dataset(keyword, df, train_index, valid_index, batch_size):
    aux = df[[c for c in columns[keyword]]]
    train_ds = dataframe_to_dataset(aux.iloc[train_index])
    valid_ds = dataframe_to_dataset(aux.iloc[valid_index])
    train_ds = train_ds.batch(batch_size)
    valid_ds = valid_ds.batch(batch_size)
    return train_ds, valid_ds

def get_input_layer(keyword, train_ds):
    encoded_inputs = []
    all_inputs = []
    for name in columns[keyword]:
        if name != 'target':
            if name in categorical:
                new_input = Input(shape=(1,), name=name, dtype="string")
                all_inputs.append(new_input)
                encoded_inputs.append(encode_categorical_feature(new_input, name, train_ds))
            else:
                new_input = Input(shape=(1,), name=name)
                encoded_inputs.append(encode_numerical_feature(new_input, name, train_ds))
                all_inputs.append(new_input)

    return concatenate(encoded_inputs), all_inputs

def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("target")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


def encode_numerical_feature(feature, name, dataset):
    normalizer = Normalization()
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))
    normalizer.adapt(feature_ds)
    encoded_feature = normalizer(feature)
    return encoded_feature

def encode_categorical_feature(feature, name, dataset):
    index = StringLookup()
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))
    index.adapt(feature_ds)
    encoded_feature = index(feature)
    encoder = CategoryEncoding(output_mode="binary")
    feature_ds = feature_ds.map(index)
    encoder.adapt(feature_ds)
    encoded_feature = encoder(encoded_feature)
    return encoded_feature

def get_class_weights(train_df):
    y_train = train_df['target']
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {0: class_weights[0],1: class_weights[1]}
    if not BALANCE_DATA:
        class_weights = {0: 1,1: 1}
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



columns = {
    'all': [
        'sex', 'anatom_site_general_challenge', 'age_approx', 
        'areas_ratio', 'major_axis_weighted_mean', 'minor_axis_weighted_mean', 'eccentricity_mean', 
        'eccentricity_std', 'R_mean', 'G_mean', 'B_mean', 'R_std', 'G_std', 'B_std', 
        'euler0', 'euler1', 'euler2', 'euler3', 'euler4', 'cnn_prediction', 'target'
    ],

    'stats': [
        'areas_ratio', 'major_axis_weighted_mean', 'minor_axis_weighted_mean', 'eccentricity_mean', 
        'eccentricity_std', 'R_mean', 'G_mean', 'B_mean', 'R_std', 'G_std', 'B_std', 'target'
    ],

    'euler': [
        'euler0', 'euler1', 'euler2', 'euler3', 'euler4', 'target'
    ],


    'stats_ext': [
        'sex', 'anatom_site_general_challenge', 'age_approx', 
        'areas_ratio', 'major_axis_weighted_mean', 'minor_axis_weighted_mean', 'eccentricity_mean', 
        'eccentricity_std', 'R_mean', 'G_mean', 'B_mean', 'R_std', 'G_std', 'B_std', 'target'
    ],

    'euler_ext': [
        'sex', 'anatom_site_general_challenge', 'age_approx', 
        'euler0', 'euler1', 'euler2', 'euler3', 'euler4', 'target'
    ],

    'stats_euler': [
        'areas_ratio', 'major_axis_weighted_mean', 'minor_axis_weighted_mean', 'eccentricity_mean', 
        'eccentricity_std', 'R_mean', 'G_mean', 'B_mean', 'R_std', 'G_std', 'B_std',
        'euler0', 'euler1', 'euler2', 'euler3', 'euler4', 'target'
    ]
}

categorical = [
    'sex', 'anatom_site_general_challenge'
]


