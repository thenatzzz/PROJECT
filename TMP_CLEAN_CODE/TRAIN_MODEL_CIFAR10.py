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

from CHECK_MODEL_DICT import format_data_without_header
from TRAIN_MODEL_MNIST import get_new_model
import numpy as np
import csv
import os
import pandas as pd

batch_size = 32
num_classes = 10
epochs = 100
data_augmentation = True
# data_augmentation = False

num_predictions = 20
saved_model_path = "CIFAR10_MODEL"
saved_model_path = "/vol/bitbucket/nj2217/CIFAR-10/"

save_dir = os.path.join(saved_model_path, 'verified_cifar10_models')
# save_dir = os.path.join(saved_model_path, 'verified_cifar10_models2')


# save_dir = os.path.join(saved_model_path, 'saved_models')
# model_name = 'keras_cifar10_trained_model.h5'

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

def save_file(file_name,each_model, scores):
    csv_columns = ['Model', '1st Layer', '2nd Layer','3rd Layer', '4th Layer','Accuracy', 'Loss']

    list_of_dict = []
    temp_dict = {}
    temp_dict['Model'] = each_model[0]
    temp_dict['1st Layer'] = each_model[1]
    temp_dict['2nd Layer'] = each_model[2]
    temp_dict['3rd Layer'] = each_model[3]
    temp_dict['4th Layer'] = each_model[4]
    temp_dict['Accuracy'] = scores[1]
    temp_dict['Loss'] = scores[0]

    list_of_dict.append(temp_dict)

    # path_to_file = "/vol/bitbucket/nj2217/CIFAR-10/"
    # folder = "501-1000/"
    # csv_file = "model_1201_1500.csv"
    # file_name = path_to_file +folder + csv_file

    try:
         with open(file_name, 'a') as csvfile:

             writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
             if os.stat(file_name).st_size == 0 : # ONLY write ROW Hedaer when file is empty
                writer.writeheader()
             for data in list_of_dict:
                 writer.writerow(data)
    except IOError:
        print("I/O error")

def load_data_cifar10():
    # The data, split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    return (x_train, y_train), (x_test, y_test)

def convert_class_vec2matrix(y_train, y_test):
    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
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
            batch_size=batch_size,
            epochs=epochs,
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
                         batch_size=batch_size),
                         epochs=epochs,
                         validation_data=(x_test, y_test),
                         callbacks=[EarlyStopping(min_delta=0.001, patience=3)],
                         workers=4)

def save_model(model,model_from_csv):
    model_name = 'keras_cifar10_'
    model_name = model_name + model_from_csv[0] + '.h5'

     # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

def get_topology_only(single_model):
    return single_model[1:-2]

def check_complete_model(single_model):
    if len(single_model) == 7:
        return True
    else:
        return False

def get_latest_model(action_array):
    file_name = "COMPLETE_CIFAR10.csv"
    from QLEARNING import open_file

    data = open_file(file_name)
    data = format_data_without_header(data)
    lastest_model = data[-1][0]
    new_model = get_new_model(lastest_model)
    print("new_model : ",new_model)
    new_action_array = [new_model]+action_array+["Unknown","Unknown"]
    return new_action_array

def train_model_cifar10(model_from_csv, single_model):
    is_complete_model = check_complete_model(single_model)

    if not is_complete_model:
        single_model = get_latest_model(single_model)

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

    if not data_augmentation:
        no_data_augmentation(model,x_train,x_test,y_train,y_test)
    else:
        data_augmentation(model,x_train,x_test,y_train,y_test)

    save_model(model,single_model)

    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

    loss = scores[0]
    accuracy = scores[1]
    print("Model ", single_model[:-2])
    file_name = "COMPLETE_CIFAR10.csv"

    save_file(file_name,single_model,scores)
    print('\n')
    clear_session()
    return accuracy

def get_data_from_csv(file_name):
    list_data = []
    with open(file_name, 'rt',encoding='utf8') as f:
        reader = csv.reader(f)
        for row in reader:
            list_data.append(row)
    return list_data[1:]
def pre_train_model_cifar10(file_name):
    data = get_data_from_csv(file_name)
    # print(data)
    # data = [['c_1','c_2','c_3','-']]
    # for index in range(len(data)):
    # print(len(data))
    for index in range(2):
        # train_model_cifar10(data,data[index])
        # list_data = [data[index]]
        # print(data[index])
        single_model = data[index]
        train_model_cifar10(data,single_model)
