from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Dropout, Dense, Input, Flatten, BatchNormalization,GaussianNoise
from keras.callbacks import TensorBoard
from pathos.loader import DatasetLoader
import tensorflow as tf
import os
from pathos.constants import SAVE_DIRECTORY


class EmotionRecognitionModel(object):
    def __init__(self, name):
        # Setting up GPU parameters
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config = config)

        self._name = name
        self._model = None

        self.dataset = None
    def build_network(self):
        self._model = Sequential()
        self._model.add(Conv2D(64, 5, activation='relu', input_shape=(48, 48, 1)))
        #self._model.add(GaussianNoise(0.01))
        self._model.add(MaxPool2D(3, strides=2))
        self._model.add(Conv2D(64, 5, activation='relu'))
        self._model.add(MaxPool2D(3, strides=2))
        self._model.add(Conv2D(128, 4, activation='relu'))
        self._model.add(Dropout(0.3))
        self._model.add(Flatten())
        self._model.add(Dense(3072, activation='relu'))
        self._model.add(Dense(7, activation='softmax'))

        print("[INFO] Compiling NN")
        self._model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
        print("[INFO] Finished building neural network")

    def load_dataset(self):
        if self.dataset is None:
            self._dataset = DatasetLoader()
        else:
            print("[WARN] Reloading dataset.")
        self._dataset.load_from_save()


    def save_model(self):
        path = os.path.join(SAVE_DIRECTORY, self._name)
        self._model.save(os.path.join(SAVE_DIRECTORY, self._name))
        print("[INFO] Model saved as [" + path + "]")

    def load_model(self):
        print("looking for network in " + os.path.join(SAVE_DIRECTORY, self._name))
        if os.path.isfile(os.path.join(SAVE_DIRECTORY, self._name)):
            self._model.load_weights(os.path.join(SAVE_DIRECTORY, self._name))
            print("[INFO] Model loaded from filesystem.")
        else:
            print("[WARNING] Model not present in filesystem.")


    def predict(self, image):
        import numpy as np
        
        if image is None:
            return None

        image = image.reshape([-1, 48, 48, 1])

        return self._model.predict(image)

    def train(self):
        self.load_dataset()
        self.build_network()
        
        # Training
        
        print('[+] Training network')

        tensorboard = TensorBoard(log_dir="logs/" + self._name,write_images=True)
        self._model.fit(self._dataset.images, self._dataset.labels, epochs=10, 
                callbacks=[tensorboard],
                validation_data=(self._dataset.images_test, self._dataset.labels_test))

        self._model.save(os.path.join(SAVE_DIRECTORY + 'test.keras'))
    
    @property
    def model(self):
        return self._model


if __name__ == '__main__':
    print("[INFO] Use main entry point to manage models.")
    pass

    
