'''
Cifar-10 classification
Original dataset and info: https://www.cs.toronto.edu/~kriz/cifar.html for more information
See: https://www.bonaccorso.eu/2016/08/06/cifar-10-image-classification-with-keras-convnet/ for further information
'''

from __future__ import print_function

import numpy as np
import csv
import os
import pandas as pd
import keras

from keras.callbacks import EarlyStopping
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.optimizers import Adam
from keras.layers.pooling import MaxPooling2D
from keras.utils import to_categorical
from keras.backend import clear_session
from keras.preprocessing.image import ImageDataGenerator

# For reproducibility
np.random.seed(1000)

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


'''
def add_conv2D(model, layer_param):
    num_filters = layer_param[0]
    size_kernel = layer_param[1]
    num_stride = layer_param[2]
    # model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(num_filters, kernel_size=(size_kernel, size_kernel),border_mode='same', activation='relu'))

def add_maxpool2D(model, layer_param):
    size_kernel = layer_param[0]
    num_stride = layer_param[1]
    model.add(MaxPooling2D(pool_size=(size_kernel, size_kernel),strides= (num_stride,num_stride),border_mode='same' ))
    model.add(Dropout(0.25))
'''
def count_model_layer(model_from_csv):
    count = 0
    for i in range(len(model_from_csv)):
        count += 1
        if model_from_csv[i] == 's':
            break
    return count

def open_file(file_csv):
    with open(file_csv) as f:
        reader = csv.reader(f, delimiter=',' , quotechar= ',' ,
                        quoting = csv.QUOTE_MINIMAL)
        data  = [r for r in reader]
        data = data[1:]      # To get rid of column name
    return data

def save_file(each_model, loss, accuracy):
    csv_columns = ['Model', '1st Layer', '2nd Layer','3rd Layer', '4th Layer','Accuracy', 'Loss']

    list_of_dict = []
    temp_dict = {}
    temp_dict['Model'] = each_model[0]
    temp_dict['1st Layer'] = each_model[1]
    temp_dict['2nd Layer'] = each_model[2]
    temp_dict['3rd Layer'] = each_model[3]
    temp_dict['4th Layer'] = each_model[4]
    temp_dict['Accuracy'] = accuracy
    temp_dict['Loss'] = loss

    list_of_dict.append(temp_dict)

    path_to_file = "/vol/bitbucket/nj2217/CIFAR-10/"

    folder = "0-250/"
    csv_file = "test.csv"

    '''
    folder = "251-500/"
    csv_file = "MODEL_dict_2.csv"

    folder = "501-750/"
    csv_file = "MODEL_dict_3.csv"

    folder = "751-1000/"
    csv_file = "MODEL_dict_4.csv"

    folder = "1001-1250/"
    csv_file = "MODEL_dict_5.csv"

    folder = "1251-1500/"
    csv_file = "MODEL_dict_6.csv"
    '''

    file = path_to_file +folder + csv_file
    try:
         with open(file, 'a') as csvfile:

             writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
             if os.stat(file).st_size == 0 : # ONLY write ROW Hedaer when file is empty
                writer.writeheader()
             for data in list_of_dict:
                 writer.writerow(data)
    except IOError:
        print("I/O error")

def train_model(model_from_csv):

    for i in range(len(model_from_csv)):
        print(model_from_csv[i][1:-2])
        # loss, accuracy = train_model(data[i][1:-2])
        #(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

        y_train = keras.utils.to_categorical(y_train, 10)
        y_test = keras.utils.to_categorical(y_test, 10)

        model = Sequential()
        num_layer = count_model_layer(model_from_csv[i][1:-2])
        print("num_layer: ",num_layer)
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))

        for index in range(num_layer):
            print("index inside train_model: ",index)
            if model_from_csv[index] == 'c_1':
                layer_param = c_1
                num_filters = layer_param[0]
                size_kernel = layer_param[1]
                num_stride = layer_param[2]
                model.add(Conv2D(num_filters, kernel_size=(size_kernel, size_kernel),border_mode='same', activation='relu'))
                # add_conv2D(model,c_1)
            elif model_from_csv[index] == 'c_2':
                layer_param = c_2
                num_filters = layer_param[0]
                size_kernel = layer_param[1]
                num_stride = layer_param[2]
                model.add(Conv2D(num_filters, kernel_size=(size_kernel, size_kernel),border_mode='same', activation='relu'))
                # add_conv2D(model,c_2)
            elif model_from_csv[index] == 'c_3':
                layer_param = c_3
                num_filters = layer_param[0]
                size_kernel = layer_param[1]
                num_stride = layer_param[2]
                model.add(Conv2D(num_filters, kernel_size=(size_kernel, size_kernel),border_mode='same', activation='relu'))
                # add_conv2D(model,c_3)
            elif model_from_csv[index] == 'c_4':
                layer_param = c_4
                num_filters = layer_param[0]
                size_kernel = layer_param[1]
                num_stride = layer_param[2]
                model.add(Conv2D(num_filters, kernel_size=(size_kernel, size_kernel),border_mode='same', activation='relu'))
                # add_conv2D(model,c_4)
            elif model_from_csv[index] == 'c_5':
                layer_param = c_5
                num_filters = layer_param[0]
                size_kernel = layer_param[1]
                num_stride = layer_param[2]
                model.add(Conv2D(num_filters, kernel_size=(size_kernel, size_kernel),border_mode='same', activation='relu'))
                # add_conv2D(model,c_5)
            elif model_from_csv[index] == 'c_6':
                layer_param = c_6
                num_filters = layer_param[0]
                size_kernel = layer_param[1]
                num_stride = layer_param[2]
                model.add(Conv2D(num_filters, kernel_size=(size_kernel, size_kernel),border_mode='same', activation='relu'))
                # add_conv2D(model,c_6)
            elif model_from_csv[index] == 'c_7':
                layer_param = c_7
                num_filters = layer_param[0]
                size_kernel = layer_param[1]
                num_stride = layer_param[2]
                model.add(Conv2D(num_filters, kernel_size=(size_kernel, size_kernel),border_mode='same', activation='relu'))
                # add_conv2D(model,c_7)
            elif model_from_csv[index] == 'c_8':
                layer_param = c_8
                num_filters = layer_param[0]
                size_kernel = layer_param[1]
                num_stride = layer_param[2]
                model.add(Conv2D(num_filters, kernel_size=(size_kernel, size_kernel),border_mode='same', activation='relu'))
                # add_conv2D(model,c_8)
            elif model_from_csv[index] == 'c_9':
                layer_param = c_9
                num_filters = layer_param[0]
                size_kernel = layer_param[1]
                num_stride = layer_param[2]
                model.add(Conv2D(num_filters, kernel_size=(size_kernel, size_kernel),border_mode='same', activation='relu'))
                # add_conv2D(model,c_9)
            elif model_from_csv[index] == 'c_10':
                layer_param = c_10
                num_filters = layer_param[0]
                size_kernel = layer_param[1]
                num_stride = layer_param[2]
                model.add(Conv2D(num_filters, kernel_size=(size_kernel, size_kernel),border_mode='same', activation='relu'))
                # add_conv2D(model,c_10)
            elif model_from_csv[index] == 'c_11':
                layer_param = c_11
                num_filters = layer_param[0]
                size_kernel = layer_param[1]
                num_stride = layer_param[2]
                model.add(Conv2D(num_filters, kernel_size=(size_kernel, size_kernel),border_mode='same', activation='relu'))
                # add_conv2D(model,c_11)
            elif model_from_csv[index] == 'c_12':
                layer_param = c_12
                num_filters = layer_param[0]
                size_kernel = layer_param[1]
                num_stride = layer_param[2]
                model.add(Conv2D(num_filters, kernel_size=(size_kernel, size_kernel),border_mode='same', activation='relu'))
                # add_conv2D(model,c_12)
            elif model_from_csv[index] == 'm_1':
                layer_param = m_1
                size_kernel = layer_param[0]
                num_stride = layer_param[1]
                model.add(MaxPooling2D(pool_size=(size_kernel, size_kernel),strides= (num_stride,num_stride),border_mode='same' ))
                model.add(Dropout(0.25))
                # add_maxpool2D(model,m_1)
            elif model_from_csv[index] == 'm_2':
                layer_param = m_2
                size_kernel = layer_param[0]
                num_stride = layer_param[1]
                model.add(MaxPooling2D(pool_size=(size_kernel, size_kernel),strides= (num_stride,num_stride),border_mode='same' ))
                model.add(Dropout(0.25))
                # add_maxpool2D(model,m_2)
            elif model_from_csv[index] == 'm_3':
                layer_param = m_3
                size_kernel = layer_param[0]
                num_stride = layer_param[1]
                model.add(MaxPooling2D(pool_size=(size_kernel, size_kernel),strides= (num_stride,num_stride),border_mode='same' ))
                model.add(Dropout(0.25))
                # add_maxpool2D(model,m_3)
            elif model_from_csv[index] == 's':
                break

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(10, activation='softmax'))

        # Compile the model
        opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

        model.compile(loss='categorical_crossentropy',
                      optimizer = opt,
                      # optimizer=Adam(lr=0.0001, decay=1e-6),
                      metrics=['accuracy'])
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        x_train /= 255
        x_test /= 255
        '''
        # Train the model
        model.fit(X_train / 255.0, to_categorical(Y_train),
                  batch_size=128,
                  # batch_size=20,
                  shuffle=True,
                  epochs=100,
                  # epochs=250,
                  validation_data=(X_test / 255.0, to_categorical(Y_test)),
                  callbacks=[EarlyStopping(min_delta=0.001, patience=3)])
        '''
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
                                         batch_size=32),
                            epochs=100,
                            validation_data=(x_test, y_test),
                            callbacks=[EarlyStopping(min_delta=0.001, patience=3)],
                            workers=4)
        '''
        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(X_train)

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(X_train/255.0, to_categorical(Y_train),
                                         batch_size=128),
                            epochs=100,
                            validation_data=(X_test/255.0, to_categorical(Y_test)),
                            callbacks=[EarlyStopping(min_delta=0.001, patience=3)],
                            workers=4)
        '''
        # Evaluate the model
        scores = model.evaluate(x_test, y_test, verbose=1)

        #scores = model.evaluate(X_test / 255.0, to_categorical(Y_test))

        print('Loss: %.3f' % scores[0])
        print('Accuracy: %.3f' % scores[1])

        loss = scores[0]
        accuracy = scores[1]

        print("Model ", model_from_csv[i][:-2])
        save_file(model_from_csv[i],loss,accuracy)
        print("MODEL ",model_from_csv[i][0], " finishes training!")
        print("Number of model trained: ",i+1)
        print('\n')
        clear_session()
if __name__ == '__main__':
    # file_name = "model.csv"
    file_name = "model.csv"

    data = open_file(file_name)
    train_model(data)

'''
    # Load the dataset
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

    # Create the model
    model = Sequential()
    num_layer = count_model_layer(model_from_csv)
    print("num_layer: ",num_layer)
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))

    for index in range(num_layer):
        print("index inside train_model: ",index)
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

    # model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)))
    # model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    # model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=0.0001, decay=1e-6),
                  metrics=['accuracy'])

    # Train the model
    model.fit(X_train / 255.0, to_categorical(Y_train),
              batch_size=128,
              shuffle=True,
              epochs=1,
              # epochs=250,
              validation_data=(X_test / 255.0, to_categorical(Y_test)),
              callbacks=[EarlyStopping(min_delta=0.001, patience=3)])

    # Evaluate the model
    scores = model.evaluate(X_test / 255.0, to_categorical(Y_test))

    print('Loss: %.3f' % scores[0])
    print('Accuracy: %.3f' % scores[1])

    loss = scores[0]
    accuracy = scores[1]

    return loss, accuracy
'''

'''
if __name__ == '__main__':
    # file_name = "model.csv"
    file_name = "model4.csv"

    data = open_file(file_name)
    train(data)
    # for i in range(len(data)):
    #     print(data[i][1:-2])
    #     loss, accuracy = train_model(data[i][1:-2])
    #     print("Model ", data[i])
    #     save_file(data[i],loss,accuracy)
    #     print("MODEL ",data[i][0], " finishes training!")
    #     print("Number of model trained: ",i+1)
    #     print('\n')
'''
