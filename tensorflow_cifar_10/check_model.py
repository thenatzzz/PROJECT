'''Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).
'''

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping
from keras.backend import clear_session

import numpy as np
import csv
import os
import pandas as pd

batch_size = 32
num_classes = 10
epochs = 1
data_augmentation = True
num_predictions = 20

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
    # model.add(MaxPooling2D(pool_size=(size_kernel, size_kernel),strides= (num_stride,num_stride) ))

    model.add(Dropout(0.25))

def count_model_layer(model_from_csv):
    count = 0
    for i in range(len(model_from_csv)):
        count += 1
        if model_from_csv[i] == 's':
            break
    return count

def train_model(model_from_csv):
    for i in range(len(model_from_csv)):
        print(model_from_csv[i][1:-2])
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

        model = Sequential()
        num_layer = count_model_layer(model_from_csv[i][1:-2])
        #print("num_layer: ",num_layer)
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))

        for index in range(1,num_layer+1):
        #    print("layer trained : ", model_from_csv[i][index])
            if model_from_csv[i][index] == 'c_1':
                add_conv2D(model,c_1)
            elif model_from_csv[i][index] == 'c_2':
                add_conv2D(model,c_2)
            elif model_from_csv[i][index] == 'c_3':
                add_conv2D(model,c_3)
            elif model_from_csv[i][index] == 'c_4':
                add_conv2D(model,c_4)
            elif model_from_csv[i][index] == 'c_5':
                add_conv2D(model,c_5)
            elif model_from_csv[i][index] == 'c_6':
                add_conv2D(model,c_6)
            elif model_from_csv[i][index] == 'c_7':
                add_conv2D(model,c_7)
            elif model_from_csv[i][index] == 'c_8':
                add_conv2D(model,c_8)
            elif model_from_csv[i][index] == 'c_9':
                add_conv2D(model,c_9)
            elif model_from_csv[i][index] == 'c_10':
                add_conv2D(model,c_10)
            elif model_from_csv[i][index] == 'c_11':
                add_conv2D(model,c_11)
            elif model_from_csv[i][index] == 'c_12':
                add_conv2D(model,c_12)
            elif model_from_csv[i][index] == 'm_1':
                add_maxpool2D(model,m_1)
            elif model_from_csv[i][index] == 'm_2':
                add_maxpool2D(model,m_2)
            elif model_from_csv[i][index] == 'm_3':
                add_maxpool2D(model,m_3)
            elif model_from_csv[i][index] == 's':
                break

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
        model.compile(loss='categorical_crossentropy',
                        optimizer=opt,
                        metrics=['accuracy'])

        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255

        if not data_augmentation:
            print('Not using data augmentation.')
            model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    shuffle=True)
        else:
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

            datagen.fit(x_train)
            model.fit_generator(datagen.flow(x_train, y_train,
                                 batch_size=batch_size),
                                 epochs=epochs,
                                 validation_data=(x_test, y_test),
                                 callbacks=[EarlyStopping(min_delta=0.001, patience=3)],
                                 workers=4)
             # Save model and weights

            # Score trained model.
            scores = model.evaluate(x_test, y_test, verbose=1)
            print('Test loss:', scores[0])
            print('Test accuracy:', scores[1])

            loss = scores[0]
            accuracy = scores[1]
            print("Model ", model_from_csv[i][:-2])
            model.summary()
            print('\n')
            clear_session()

def create_model():
    final_list = []
    count = 1
    max_pooling = ['m_3','m_2','m_1']
    # c2 = ['c_2']
    tmp_list = []
    for m1 in range(len(max_pooling)):
#        tmp_list = []
        for m2 in range(len(max_pooling)):
            for m3 in range(len(max_pooling)):
                for m4 in range(len(max_pooling)):
                    model_name = "model_"+str(count)
                    tmp_list.append(model_name)
                    tmp_list.append(max_pooling[m1])
                    tmp_list.append(max_pooling[m2])
                    tmp_list.append(max_pooling[m3])
                    tmp_list.append(max_pooling[m4])
                    tmp_list.append("unknown")
                    tmp_list.append("unknown")
                    print(tmp_list)
                    final_list.append(tmp_list)
                    tmp_list = []

    return final_list
if __name__ == '__main__':
    data = create_model()
    print(data)
    # try:
    train_model(data)
    # except ValueError:
        # print("ERROR")
    # train_model(data)
