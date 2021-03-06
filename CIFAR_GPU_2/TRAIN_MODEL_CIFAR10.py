from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.backend import clear_session

from HELPER_FUNCTION import format_data_without_header,get_data_from_csv, \
                            check_complete_model, get_topology_only,count_model_layer,\
                            get_latest_model_list, save_trained_model_in_csv,\
                            get_current_model_number, get_new_model_number
import numpy as np
import csv
import os
import pandas as pd

MAIN_FILE = "COMPLETE_CIFAR10.csv"
BATCH_SIZE = 32
NUM_CLASSES = 10
EPOCHS = 100
DATA_AUGMENTATION = True

NUM_PREDICTIONS = 20
SAVED_MODEL_PATH = "/vol/bitbucket/nj2217/CIFAR-10/"

SAVE_DIR = os.path.join(SAVED_MODEL_PATH, 'test_verified_model')

#PREDEFINED LAYER
# 1. Convolutional Layer [number of output filter, kernel size, stride]
c_1 = [32,3,1]
c_2 = [32,4,1]
c_3 = [32,5,1]
c_4 = [36,3,1]
c_5 = [36,4,1]
c_6 = [36,5,1]
c_7 = [48,3,1]
c_8 = [48,4,1]
c_9 = [48,5,1]
c_10 = [64,3,1]
c_11 = [64,4,1]
c_12 = [64,5,1]
# 2. Pooling Layer [kernel size, stride]
m_1 = [2,2]
m_2 = [3,2]
m_3 = [5,3]
# 3. Softmax Layer (Termination Layer)
s = [0]

def add_conv2D(model, layer_param):
    num_filters = layer_param[0]
    size_kernel = layer_param[1]
    num_stride = layer_param[2]
    model.add(Conv2D(num_filters, kernel_size=(size_kernel, size_kernel),border_mode='same', activation='relu'))

def add_maxpool2D(model, layer_param):
    size_kernel = layer_param[0]
    num_stride = layer_param[1]
    model.add(MaxPooling2D(pool_size=(size_kernel, size_kernel),strides= (num_stride,num_stride),border_mode='same' ))
    model.add(Dropout(0.25))

def load_data_cifar10():
    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    return (x_train, y_train), (x_test, y_test)

def convert_class_vec2matrix(y_train, y_test):
    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
    return y_train, y_test

def cnn_model_fn(model,num_layer,model_from_csv):
    print("model_from_csv: ",model_from_csv)
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
    print("num_layer: ", num_layer)
    for index in range(1,num_layer+1):
        print("index : ", index)
        if model_from_csv[index] == 'c_1':
            add_conv2D(model,c_1)
        elif model_from_csv[index] == 'c_2':
            add_conv2D(model,c_2)
        elif model_from_csv[index] == 'c_3':
            add_conv2D(model,c_3)
        elif model_from_csv[index] == 'c_4':
            add_conv2D(model,c_4)
        elif model_from_csv[index] == 'c_5':
            add_conv2D(model,c_5)
        elif model_from_csv[index] == 'c_6':
            add_conv2D(model,c_6)
        elif model_from_csv[index] == 'c_7':
            add_conv2D(model,c_7)
        elif model_from_csv[index] == 'c_8':
            add_conv2D(model,c_8)
        elif model_from_csv[index] == 'c_9':
            add_conv2D(model,c_9)
        elif model_from_csv[index] == 'c_10':
            add_conv2D(model,c_10)
        elif model_from_csv[index] == 'c_11':
            add_conv2D(model,c_11)
        elif model_from_csv[index] == 'c_12':
            add_conv2D(model,c_12)
        elif model_from_csv[index] == 'm_1':
            add_maxpool2D(model,m_1)
        elif model_from_csv[index] == 'm_2':
            add_maxpool2D(model,m_2)
        elif model_from_csv[index] == 'm_3':
            add_maxpool2D(model,m_3)
        elif model_from_csv[index] == 's':
            break

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    return model

def format_data(x_train,x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    return x_train, x_test

def no_data_augmentation(model,x_train,x_test,y_train,y_test):
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(x_test, y_test),
            callbacks=[EarlyStopping(min_delta=0.001, patience=3)],
            shuffle=True)

def data_augmentation(model,x_train,x_test,y_train,y_test):
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        fill_mode='nearest',  # set mode for filling points outside the input boundaries
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        rescale=None,  # set rescaling factor (applied before any other transformation)
        preprocessing_function=None,  # set function that will be applied on each input
        data_format=None,  # image data format, either "channels_first" or "channels_last"
        validation_split=0.0)  # fraction of images reserved for validation (strictly between 0 and 1)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                         batch_size=BATCH_SIZE),
                         epochs=EPOCHS,
                         validation_data=(x_test, y_test),
                         callbacks=[EarlyStopping(min_delta=0.001, patience=3)],
                         workers=4)

def save_model_keras_cifar10(model,model_from_csv):
    model_name = 'keras_cifar10_'
    model_name = model_name + model_from_csv[0] + '.h5'

     # Save model and weights
    if not os.path.isdir(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    model_path = os.path.join(SAVE_DIR, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

def train_model_cifar10( single_model, is_verify = False):
    file = MAIN_FILE
    is_complete_model = check_complete_model(single_model)

    if not is_complete_model:
        single_model = get_latest_model_list(single_model,file)
        model_name = single_model[0]
        cur_model_num = get_current_model_number(model_name)
        model_index = get_new_model_number(cur_model_num)

    print("single_model: ",single_model)
    (x_train, y_train), (x_test, y_test) = load_data_cifar10()
    y_train, y_test = convert_class_vec2matrix(y_train,y_test)

    model = Sequential()
    tmp_single_model = get_topology_only(single_model)
    num_layer = count_model_layer(tmp_single_model)

    model = cnn_model_fn(model,num_layer,single_model)

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])

    x_train, x_test = format_data(x_train,x_test)

    if not DATA_AUGMENTATION:
        no_data_augmentation(model,x_train,x_test,y_train,y_test)
    else:
        data_augmentation(model,x_train,x_test,y_train,y_test)

    if not is_verify:
        save_model_keras_cifar10(model,single_model)

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    loss = scores[0]
    accuracy = scores[1]
    print("Model ", single_model)

    if not is_verify:
        save_trained_model_in_csv(file,single_model,scores)
    print('\n')
    clear_session()
    return accuracy

def pre_train_model_cifar10(file_name, output_file_name):
    global MAIN_FILE
    MAIN_FILE = output_file_name

    data = get_data_from_csv(file_name)
    data = format_data_without_header(data)

    for index in range(len(data)):
        single_model = data[index]
        train_model_cifar10(single_model)
