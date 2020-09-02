import numpy as np
from tensorflow import keras
from tensorflow.keras import layers # pylint: disable= import-error

class MelanomaDetector:

    num_classes = 2
    input_shape = (64, 64, 1)

    def __init__(self, train_images, train_labels, batch_size = 128, epochs = 15):
        # train_images, train_labels must be:
        #  (num_images, image.shape[0], image.shape[1]) shapes arrays
        #  i.e. images must be aggregated into one 3D image


        # Make sure images have the desired shape
        train_images = np.expand_dims(train_images, -1)
        print("train_images shape:", train_images.shape)
        print(train_images.shape[0], "train samples")

        # convert class vectors to binary class matrices (converto i -> [0...1..0] where the 1 is in the i-th position)
        train_labels = keras.utils.to_categorical(train_labels, MelanomaDetector.num_classes)

        model = keras.Sequential(
            [
                keras.Input(shape=MelanomaDetector.input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(MelanomaDetector.num_classes, activation="softmax"),
            ]
        )

        model.summary()

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        model.fit(train_images, train_labels, batch_size=batch_size, epochs=epochs, validation_split=0.1)

        self.model = model

    def evaluate(self, test_images, test_labels):
        test_images = np.expand_dims(test_images, -1)
        print(test_images.shape[0], "test samples")
        test_labels = keras.utils.to_categorical(test_labels, MelanomaDetector.num_classes)

        score = self.model.evaluate(test_images, test_labels, verbose=0)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

