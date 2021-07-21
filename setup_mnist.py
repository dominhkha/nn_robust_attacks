## setup_mnist.py -- mnist data and model loading code
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.

import tensorflow as tf
import numpy as np
import os
import pickle
import gzip
import urllib.request

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import np_utils
from tensorflow.keras.models import load_model

def extract_data(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(num_images*28*28)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data / 255.
        data = data.reshape(num_images, 28, 28, 1)
        return data

def extract_labels(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * num_images)
        labels = np.frombuffer(buf, dtype=np.uint8)
    return (np.arange(10) == labels[:, None]).astype(np.float32)

class MNIST:
    def __init__(self):
        if not os.path.exists("data"):
            os.mkdir("data")
            # files = ["train-images-idx3-ubyte.gz",
            #          "t10k-images-idx3-ubyte.gz",
            #          "train-labels-idx1-ubyte.gz",
            #          "t10k-labels-idx1-ubyte.gz"]
            # for name in files:
            #
            #     urllib.request.urlretrieve('http://yann.lecun.com/exdb/mnist/' + name, "data/"+name)

        # train_data = extract_data("data/train-images-idx3-ubyte.gz", 60000)
        # train_labels = extract_labels("data/train-labels-idx1-ubyte.gz", 60000)
        # self.test_data = extract_data("data/t10k-images-idx3-ubyte.gz", 10000)
        # self.test_labels = extract_labels("data/t10k-labels-idx1-ubyte.gz", 10000)


        (train_data, train_labels), (self.test_data, self.test_labels) = tf.keras.datasets.mnist.load_data()

        train_labels = (np.arange(10) == train_labels[:, None]).astype(np.float32)
        self.test_labels = (np.arange(10) == self.test_labels[:, None]).astype(np.float32)

        train_data = train_data.astype(np.float32) / 255
        self.test_data = self.test_data.astype(np.float32)

        VALIDATION_SIZE = 1000

        train_data = train_data.reshape(60000, 28, 28, 1)
        self.test_data = self.test_data.reshape(10000, 28, 28, 1)
        self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
        self.validation_labels = train_labels[:VALIDATION_SIZE]
        self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
        self.train_labels = train_labels[VALIDATION_SIZE:]


class MNISTModel:
    def __init__(self, restore, session=None):
        self.num_channels = 1
        self.image_size = 28
        self.num_labels = 10

        # model = Sequential()

        # model.add(Conv2D(32, (3, 3),
        #                  input_shape=(28, 28, 1)))
        # model.add(Activation('relu'))
        # model.add(Conv2D(32, (3, 3)))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))

        # model.add(Conv2D(64, (3, 3)))
        # model.add(Activation('relu'))
        # model.add(Conv2D(64, (3, 3)))
        # model.add(Activation('relu'))
        # model.add(MaxPooling2D(pool_size=(2, 2)))

        # model.add(Flatten())
        # model.add(Dense(200))
        # model.add(Activation('relu'))
        # model.add(Dense(200))
        # model.add(Activation('relu'))
        # model.add(Dense(10))
        # model.load_weights(restore)

        # model = Sequential()
        # model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
        # model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
        # model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Flatten())
        # model.add(Dense(200, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(200, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(10, activation="relu"))
        # model.load_weights(restore)
        self.model = tf.keras.models.load_model('models/lenet_v21.h5', compile=False)
        self.model = tf.keras.models.Model(inputs=self.model.input,
                                                   outputs=self.model.get_layer('dense_2').output)

        # with open('models/Lenet_v2.json') as json_file:
        #     json_config = json_file.read()
        #     new_model = tf.keras.models.model_from_json(json_config, custom_objects={'AttentionLayer': Functional})
        # new_model.load_weights('models/Lenet_v2_weights.h5')
        # self.model = new_model

    def predict(self, data):
        return self.model(data)
