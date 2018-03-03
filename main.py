from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import scipy.io as spio
import keras
from keras.layers import Activation, BatchNormalization, Dense, Dropout
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import normalize, np_utils

DATA = 'data4students.mat'


def main():
    data = spio.loadmat(DATA, squeeze_me=True)

    x_train = data['datasetInputs'][0]
    y_train = data['datasetTargets'][0]

    x_val = data['datasetInputs'][1]
    y_val = data['datasetTargets'][1]

    x_test = data['datasetInputs'][2]
    y_test = data['datasetTargets'][2]

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    x_val = np.array(x_val)
    y_val = np.array(y_val)

    normalize_input(x_train)
    normalize_input(x_val)
    normalize_input(x_test)

    activation_fns = ['relu', 'relu', 'relu']
    layers = ['900', '300', '100']
    learning_rate = '0.01'
    decay_lr = '0.0'
    momentum = '0.5'
    nesterov = 'True'
    print(activation_fns)
    print(layers)
    print('learning rate: ' + learning_rate)
    print('decay rate: ' + decay_lr)
    print('momentum: ' + momentum)
    print('Nesterov: ' + nesterov)

    model = Sequential()

    model.add(Dense(int(layers[0]), input_dim=900, activation=activation_fns[0]))
    for i in range(1, len(layers)):
        # model.add(Dropout(0.5))
        model.add(Dense(int(layers[i]), activation=activation_fns[i]))
    model.add(Dense(7, activation="softmax"))

    print("[INFO] compiling model...")
    sgd = SGD(lr=float(learning_rate), momentum=float(momentum), decay=float(decay_lr), nesterov=bool(nesterov))

    model.compile(loss="categorical_crossentropy",
                            optimizer=sgd, metrics=["accuracy"])

    early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_acc',
                            min_delta=0.01, patience=10, verbose=1, mode='max')

    log_dir = "./Logs/" + str.join('_', activation_fns + layers + ['lr', learning_rate, 'dr', decay_lr, 'm', momentum])
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0,
                              write_graph=True, write_images=True)
    model.fit(x_train, y_train, epochs=80, batch_size=128,
    validation_data=(x_val, y_val), callbacks=[tensorboard, early_stop_callback])

    # show the accuracy on the testing set
    print("[INFO] evaluating on testing set...")
    (loss, accuracy) = model.evaluate(x_test, y_test,
    batch_size=128, verbose=1)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
    accuracy * 100))

def normalize_input(x):
    np.transpose(x)
    for i in range(len(x)):
        avg = np.average(x[i])
        for j in range(len(x[i])):
            if avg != 0:
                x[i][j] = (x[i][j] - avg) / avg
    np.transpose(x)
    return x

if __name__ == "__main__":
    main()
