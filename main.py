from __future__ import print_function

import argparse
import os
import sys
import datetime
import multiprocessing

from lr_schedule_fns import decay_constant, decay_scaling_factor, decay_after_constant, get_learning_rate, get_lr_param
import numpy as np
import scipy.io as spio
import keras
from keras.layers import Activation, BatchNormalization, Dense, Dropout
from keras.callbacks import TensorBoard
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import normalize, np_utils
from sklearn.metrics import confusion_matrix

DATA = 'data4students.mat'
LOG_DIR = './Logs/'

activation_fns = ['relu', 'relu', 'relu']

layers = [900, 300, 100]
dropouts = [0.0, 0.1, 0.0]

learning_rate = get_learning_rate()
momentum = 0.5
decay_lr = 0.0
nesterov = True
early_stop_min_delta = 0.01
patience = 20

lr_param = get_lr_param()
lr_scheduler_fn = decay_scaling_factor
callbacks = []
tensorboard = True
normalize_imgs = True
max_epochs = 80
batch_size = 128

normalize_cm = False
def main():
    log_dir = LOG_DIR
    log_dir = init_log_dir(log_dir)

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

    model = Sequential()

    model.add(Dense(layers[0], input_dim=900, activation=activation_fns[0]))
    model.add(Dropout(dropouts[0]))
    for i in range(1, len(layers)):
        model.add(Dropout(dropouts[i]))
        model.add(Dense(layers[i], activation=activation_fns[i]))
    model.add(Dense(7, activation='softmax'))

    print("[INFO] compiling model...")
    sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_lr, nesterov=nesterov)

    model.compile(loss='categorical_crossentropy',
                            optimizer=sgd, metrics=['accuracy'])

    early_stop_callback = keras.callbacks.EarlyStopping(monitor='val_acc',
                            min_delta=early_stop_min_delta, patience=patience, verbose=1, mode='max')
    append_to_callbacks(early_stop_callback)

    if lr_scheduler_fn:
        learning_rate_scheduler = keras.callbacks.LearningRateScheduler(
                                                    lr_scheduler_fn, verbose=0)
        append_to_callbacks(learning_rate_scheduler)
        log_dir = append_params_to_log_dir(log_dir, ['lr_scheduler', lr_scheduler_fn.__name__])
    else:
        log_dir = append_params_to_log_dir(log_dir, ['lr_scheduler', lr_scheduler_fn])

    log_dir = check_unique_log_dir(log_dir)

    if tensorboard:
        tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=0,
                                  write_graph=True, write_images=True)
        append_to_callbacks(tensorboard_cb)

    print("-------------------------------------------------------------------")
    print("Log Directory: " + log_dir)
    print("-------------------------------------------------------------------")

    if normalize_imgs:
        print("[INFO] Normalizing Images..")
        normalize_input(x_train)
        normalize_input(x_val)
        normalize_input(x_test)

    model.fit(x_train, y_train, epochs=max_epochs, batch_size=batch_size,
                        validation_data=(x_val, y_val), callbacks=callbacks)

    # show the accuracy on the testing set
    print("[INFO] evaluating on testing set...")
    (loss, accuracy) = model.evaluate(x_test, y_test,
                                            batch_size=batch_size, verbose=1)
    print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
                                                        accuracy * 100))
    y_pred = model.predict(x_test, batch_size=batch_size, verbose=1)

    y_pred = y_pred.argmax(1)
    y_true = y_test.argmax(1)

    conf_matrix = confusion_matrix(y_true, y_pred)

    print_confusion_matrix(conf_matrix, normalize_cm)

def normalize_input(x):
    pool = multiprocessing.Pool(8)
    for i in range(len(x)):
        x[i] = pool.map(scale,[x[i]])[0]
    return x

def append_params_to_log_dir(log_dir, params):
    log_dir += str.join('_', [str(i) for i in params])
    log_dir += '_'
    return log_dir

def init_log_dir(log_dir):
    log_dir = append_params_to_log_dir(log_dir, activation_fns + layers)
    log_dir = append_params_to_log_dir(log_dir, ['lr', learning_rate])
    log_dir = append_params_to_log_dir(log_dir, ['dr', decay_lr])
    log_dir = append_params_to_log_dir(log_dir, ['m', momentum])
    log_dir = append_params_to_log_dir(log_dir, ['nest', nesterov])
    log_dir = append_params_to_log_dir(log_dir, ['patience', patience])
    log_dir = append_params_to_log_dir(log_dir, ['lr_param', lr_param])
    log_dir = append_params_to_log_dir(log_dir, ['dropouts',] + dropouts)
    return log_dir

def append_to_callbacks(callback):
    callbacks.append(callback)

def check_unique_log_dir(log_dir):
    if os.path.isdir(log_dir):
        print("Log Directory already exists! Appending date to log dir.")
        log_dir += datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return log_dir

def print_confusion_matrix(cm, normalize_cm):
    if normalize_cm:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

def scale(img):
    avg = np.average(img)
    for i in range(len(img)):
        if avg != 0:
            img[i] = (img[i] - avg) / avg
    return img

if __name__ == "__main__":
    main()
