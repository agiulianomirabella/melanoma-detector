import numpy as np
from tensorflow import keras
from tensorflow.keras import layers # pylint: disable= import-error
from tensorflow.keras.preprocessing.image import ImageDataGenerator # pylint: disable= import-error

#from https://www.kaggle.com/pytorch/densenet169

num_classes = 2
input_shape = (128, 128, 1)
path = '../../input-data/jpeg/'

batch_size = 32
epochs = 15

image_datagen = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True)

def createModel(train_df):
    # train_images, train_labels must be:
    #  (num_images, image.shape[0], image.shape[1]) shapes arrays
    #  i.e. images must be aggregated into one 3D image

    image_height = 32
    image_width  = 32
    batch_size   = 32

    # Carregando as imagens de treinamento com ImageDataGenerator
    image_datagen = ImageDataGenerator(rescale = 1./255,          # normaliza valores dos pixels da imagem entre 0-1
                                    validation_split = 0.3)    # divide os dados do dataset em uma proporção de treinamento e validação


    train_generator = image_datagen.flow_from_dataframe(dataframe=train_df,                   # dataframe com dados da imagem
                                                        directory=path + "train",                      # diretório com as imagens
                                                        x_col="image_name",                     # nome da coluna do dataframe com os nomes das imagens
                                                        y_col="benign_malignant",               # nome da coluna do dataframe com a especificação das classes
                                                        class_mode="binary",                    # modo binário, irá selecionar as imagens em duas classes
                                                        target_size=(image_height,image_width), # tamanho das imagens de treinamento
                                                        batch_size=batch_size,                  # quantidade de imagens por pacote
                                                        subset="training",                      # subset de treinamento ou validação
                                                        color_mode="grayscale")                       # modo de carregamento da imagem em 3 canais RGB

    validation_generator = image_datagen.flow_from_dataframe(dataframe=train_df,
                                                            directory=path + "train",
                                                            x_col="image_name",
                                                            y_col="benign_malignant",
                                                            class_mode="binary",
                                                            target_size=(image_height,image_width),
                                                            batch_size=batch_size,
                                                            subset="validation",
                                                            color_mode="grayscale")
    ks3 = (3, 3)
    ks5 = (5, 5)
    ks7 = (7, 7)

    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=ks7, activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dense(16, activation='relu'),

            layers.Conv2D(64, kernel_size=ks5, activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dense(32, activation='relu'),

            layers.Conv2D(128, kernel_size=ks3, activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),

            layers.Flatten(),
            layers.Dense(1024, activation="relu"),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])

    model.summary()

    STEP_SIZE_TRAIN = train_generator.n//train_generator.batch_size
    STEP_SIZE_VALID = validation_generator.n//validation_generator.batch_size

    model.fit(train_generator,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_data=validation_generator,
            validation_steps=STEP_SIZE_VALID,
            epochs=50)

    return model

def testModel(model, test_df):

    test_generator = image_datagen.flow_from_dataframe(
        test_df,
        directory=path + 'train',
        x_col="image_name",
        y_col="benign_malignant",
        target_size=input_shape[:2],
        color_mode='grayscale',
        class_mode="binary",
        batch_size=batch_size,
        shuffle=True,
    )

    score = model.evaluate(test_generator, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    print(len(score))

    return score