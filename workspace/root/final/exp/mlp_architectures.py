import tensorflow as tf
from tensorflow.keras.models import Sequential # pylint: disable= import-error
from tensorflow.keras.layers import Dense, Dropout # pylint: disable= import-error
from tensorflow.keras import Input, Model, regularizers # pylint: disable= import-error
from root.final.exp.mlp_utils import get_input_layer

METRICS     = ['AUC', 'accuracy']



def get_all_architecture(input_layer):

    x = Dense(16, activation='tanh', kernel_regularizer= regularizers.l1_l2(l1=0.01, l2=0.01))(input_layer)
    x = Dropout(0.5)(x)
    x = Dense(12, activation='tanh', kernel_regularizer= regularizers.l1_l2(l1=0.01, l2=0.01))(x)
    x = Dropout(0.5)(x)
    x = Dense(10, activation='tanh', kernel_regularizer= regularizers.l1_l2(l1=0.01, l2=0.01))(x)
    x = Dropout(0.5)(x)
    x = Dense(8, activation='tanh', kernel_regularizer= regularizers.l1_l2(l1=0.01, l2=0.01))(x)
    x = Dropout(0.5)(x)
    x = Dense(4, activation='tanh', kernel_regularizer= regularizers.l1_l2(l1=0.01, l2=0.01))(x)
    x = Dropout(0.25)(x)
    x = Dense(2, activation='tanh', kernel_regularizer= regularizers.l1_l2(l1=0.01, l2=0.01))(x)
    #x = Dropout(0.5)(x)
    #x = Dense(4, activation='relu', kernel_regularizer= regularizers.l1_l2(l1=0.001, l2=0.001))(x)
    output = Dense(1, activation="sigmoid")(x)

    return output





def get_stats_architecture(input_layer):

    x = Dense(12, activation="relu")(input_layer)
    x = Dense(8, activation='relu', kernel_regularizer= regularizers.l1_l2(l1=0.001, l2=0.001))(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)

    return output

def get_euler_architecture(input_layer):

    x = Dense(8, activation="relu")(input_layer)
    x = Dense(4, activation='relu', kernel_regularizer= regularizers.l1_l2(l1=0.001, l2=0.001))(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid")(x)

    return output

def get_stats_ext_architecture(input_layer):

    x = Dense(16, activation="relu")(input_layer)
    x = Dense(8, activation='relu', kernel_regularizer= regularizers.l1_l2(l1=0.001, l2=0.001))(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)

    return output

def get_euler_ext_architecture(input_layer):

    x = Dense(16, activation="relu")(input_layer)
    x = Dense(8, activation='relu', kernel_regularizer= regularizers.l1_l2(l1=0.001, l2=0.001))(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)

    return output

def get_stats_euler_architecture(input_layer):

    x = Dense(16, activation="relu")(input_layer)
    x = Dense(8, activation='relu', kernel_regularizer= regularizers.l1_l2(l1=0.001, l2=0.001))(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation="sigmoid")(x)

    return output


def get_model(keyword, train_data):
    input_layer, all_inputs = get_input_layer(keyword, train_data)
    if keyword == 'all':
        output = get_all_architecture(input_layer)
    if keyword == 'stats':
        output = get_stats_architecture(input_layer)
    if keyword == 'euler':
        output = get_euler_architecture(input_layer)
    if keyword == 'stats_ext':
        output = get_stats_ext_architecture(input_layer)
    if keyword == 'euler_ext':
        output = get_euler_ext_architecture(input_layer)
    if keyword == 'stats_euler':
        output = get_stats_euler_architecture(input_layer)

    model = Model(all_inputs, output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=METRICS)
    #model.summary()
    
    return model
    
