## train_models.py -- train the neural network models for attacking
##
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## This program is licenced under the BSD 2-Clause licence,
## contained in the LICENCE file in this directory.


import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, InputLayer, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD

import tensorflow as tf
from setup_mnist import MNIST
from setup_cifar import CIFAR
import os

def train(data, file_name, params, num_epochs=50, batch_size=128, train_temp=1, init=None):
    """
    Standard neural network training procedure.
    """
    # model = Sequential()

    # print(data.train_data.shape)

    # model.add(Conv2D(params[0], (3, 3),
    #                         input_shape=data.train_data.shape[1:]))
    # model.add(Activation('relu'))
    # model.add(Conv2D(params[1], (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Conv2D(params[2], (3, 3)))
    # model.add(Activation('relu'))
    # model.add(Conv2D(params[3], (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Flatten())
    # model.add(Dense(params[4]))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(params[5]))
    # model.add(Activation('relu'))
    # model.add(Dense(10))
    model = Sequential()
    model.add(InputLayer(input_shape=data.train_data.shape[1:]))

    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, kernel_size=(11, 11),
                    strides=(4, 4), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(5, 5),
                    padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3),
                    padding='same', activation='relu'))
    model.add(BatchNormalization())

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3),
                    padding='same', activation='relu'))
    model.add(BatchNormalization())

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3),
                    padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2),
                          padding='same'))

    #Passing into a Fully Connected layer
    model.add(Flatten())

    # 1st Fully Connected Layer
    model.add(Dense(units=4096,activation='relu'))
    model.add(Dropout(0.4))

    # 2nd Fully Connected Layer
    model.add(Dense(units=4096, activation='relu'))
    model.add(Dropout(0.4))

    # 3rd Fully Connected Layer
    model.add(Dense(units=1000, activation='relu'))
    model.add(Dropout(0.4))

    # 4th Output Layer
    model.add(Dense(units=10, activation='softmax'))

    if init != None:
        model.load_weights(init)

    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted/train_temp)

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.fit(data.train_data, data.train_labels,
              batch_size=batch_size,
              validation_data=(data.validation_data, data.validation_labels),
              epochs=1,
              shuffle=True)


    if file_name != None:
        model.save(file_name)

    return model

def train_distillation(data, file_name, params, num_epochs=50, batch_size=128, train_temp=1):
    """
    Train a network using defensive distillation.

    Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks
    Nicolas Papernot, Patrick McDaniel, Xi Wu, Somesh Jha, Ananthram Swami
    IEEE S&P, 2016.
    """
    if not os.path.exists(file_name+"_init"):
        # Train for one epoch to get a good starting point.
        train(data, file_name+"_init", params, 1, batch_size)

    # now train the teacher at the given temperature
    teacher = train(data, file_name+"_teacher", params, num_epochs, batch_size, train_temp,
                    init=file_name+"_init")

    # evaluate the labels at temperature t
    predicted = teacher.predict(data.train_data)
    with tf.Session() as sess:
        y = sess.run(tf.nn.softmax(predicted/train_temp))
        print(y)
        data.train_labels = y

    # train the student model at temperature t
    student = train(data, file_name, params, num_epochs, batch_size, train_temp,
                    init=file_name+"_init")

    # and finally we predict at temperature 1
    predicted = student.predict(data.train_data)

    print(predicted)

if not os.path.isdir('models'):
    os.makedirs('models')

# train(CIFAR(), "models/cifar", [64, 64, 128, 128, 256, 256], num_epochs=50)
train(MNIST(), "models/mnist", [32, 32, 64, 64, 200, 200], num_epochs=10)

# train_distillation(MNIST(), "models/mnist-distilled-100", [32, 32, 64, 64, 200, 200],
                  #  num_epochs=30, train_temp=100)
# train_distillation(CIFAR(), "models/cifar-distilled-100", [64, 64, 128, 128, 256, 256],
                  #  num_epochs=50, train_temp=100)
