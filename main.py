'''Train a simple deep CNN on the CIFAR10 small images dataset.

It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import scipy.io as spio
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

    x_train = np.array(x_train) /255.0
    y_train = np.array(y_train)
    x_test = np.array(x_test) /255.0
    y_test = np.array(y_test)
    x_val = np.array(x_val) /255.0
    y_val = np.array(y_val)
    #
    # normalize_input(x_train)
    # normalize_input(x_val)
    # normalize_input(x_test)

    x_train = normalize(x_train, axis=1, order=2)
    x_val = normalize(x_val, axis=1, order=2)
    x_test = normalize(x_test, axis=1, order=2)

    model = Sequential()

    model.add(Dense(700, input_dim=900, activation="relu"))
    model.add(Dense(500, activation="relu"))
    model.add(Dense(250, activation="relu"))
    model.add(Dense(150, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(50, activation="relu"))
    model.add(Dense(7, activation="softmax"))


    print("[INFO] compiling model...")
    sgd = SGD(lr=0.01, momentum=0.5)
    model.compile(loss="categorical_crossentropy",
    optimizer=sgd, metrics=["accuracy"])
    log_dir_name = sys.argv[1]
    log_dir = "./Logs/" + log_dir_name
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=0,
                              write_graph=True, write_images=True)
    early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=0.05, patience=5, verbose=1, mode='max')
    model.fit(x_train, y_train, epochs=40, batch_size=128,
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
