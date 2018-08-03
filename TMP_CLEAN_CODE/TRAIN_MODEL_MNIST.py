from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from QLEARNING import open_file
from CHECK_MODEL_DICT import format_data_without_header


import numpy as np
import tensorflow as tf
import csv
import os
import pandas as pd

tf.logging.set_verbosity(tf.logging.INFO)

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


data1 = []
global_index = 0
model_index = 0

def count_model_layer(model):
    print("model : ",model)
    counter = 1
    list_length =  len(model)
    while  list_length > 1 :
        if model[counter] == '-' :
            break
        list_length -= 1
        counter += 1
    return counter-1

def make_conv2d(input_layer,layer_param):
    num_filters = layer_param[0]
    size_kernel = layer_param[1]
    num_stride = layer_param[2]
    return tf.layers.conv2d(
            inputs = input_layer,
            filters = num_filters,
            kernel_size = [size_kernel,size_kernel],
            padding = "same",
            activation= tf.nn.relu)

def make_pool2d(input_layer,layer_param):
    size_kernel = layer_param[0]
    num_stride = layer_param[1]
    return tf.layers.max_pooling2d(inputs= input_layer,
                                    pool_size=[size_kernel,size_kernel],
                                    strides=num_stride,
                                    padding= "SAME")

def cnn_model_fn_2(features,labels, mode):
    input_layer = tf.reshape(features["x"], [-1,28,28,1])
    num_layer = count_model_layer(data1[global_index][:-2])

    layer = input_layer
    temp_layer = 0
    for index  in range(1,num_layer):

        if data1[global_index][index]== 'c_1':
            temp_layer = make_conv2d(layer, c_1)
        elif data1[global_index][index] == 'c_2':
            temp_layer = make_conv2d(layer, c_2)
        elif data1[global_index][index] == 'c_3':
            temp_layer = make_conv2d(layer, c_3)
        elif data1[global_index][index] == 'c_4':
            temp_layer = make_conv2d(layer, c_4)
        elif data1[global_index][index] == 'c_5':
            temp_layer = make_conv2d(layer, c_5)
        elif data1[global_index][index] == 'c_6':
            temp_layer = make_conv2d(layer, c_6)
        elif data1[global_index][index] == 'c_7':
            temp_layer = make_conv2d(layer, c_7)
        elif data1[global_index][index] == 'c_8':
            temp_layer = make_conv2d(layer, c_8)
        elif data1[global_index][index] == 'c_9':
            temp_layer = make_conv2d(layer, c_9)
        elif data1[global_index][index] == 'c_10':
            temp_layer = make_conv2d(layer, c_10)
        elif data1[global_index][index] == 'c_11':
            temp_layer = make_conv2d(layer, c_11)
        elif data1[global_index][index] == 'c_12':
            temp_layer = make_conv2d(layer, c_12)
        elif data1[global_index][index] == 'm_1':
            temp_layer = make_pool2d(layer, m_1)
        elif data1[global_index][index] == 'm_2':
            temp_layer = make_pool2d(layer, m_2)
        elif data1[global_index][index] == 'm_3':
            temp_layer = make_pool2d(layer, m_3)
        elif data1[global_index][index] == 's':
            break
        layer = temp_layer

    shape_array = layer.get_shape()
    pool2_flat = tf.reshape(layer, [-1, shape_array[1] * shape_array[2] * shape_array[3]])

    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

    logits = tf.layers.dense(inputs=dropout, units=10)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
      train_op = optimizer.minimize(
          loss=loss,
          global_step=tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def save_to_file(file_name,eval_results):

    csv_columns = ['Model', '1st Layer', '2nd Layer','3rd Layer', '4th Layer','Accuracy', 'Loss']
    list_of_dict = []
    temp_dict = {}
    temp_dict['Model'] = data1[global_index][0]
    temp_dict['1st Layer'] = data1[global_index][1]
    temp_dict['2nd Layer'] = data1[global_index][2]
    temp_dict['3rd Layer'] = data1[global_index][3]
    temp_dict['4th Layer'] = data1[global_index][4]
    temp_dict['Accuracy'] = eval_results['accuracy']
    temp_dict['Loss'] = eval_results['loss']
    list_of_dict.append(temp_dict)

    print("\n####################### FINISIH TRAINING MODEL: ",temp_dict['Model'], " : #########################")
    print('\n\n\n')

    try:
       with open(file_name, 'a') as csvfile:
           writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
           if os.stat(file_name).st_size == 0 : # ONLY write ROW Hedaer when file is empty
              writer.writeheader()
           for data in list_of_dict:
               writer.writerow(data)
    except IOError:
       print("I/O error")

def check_format(action_array):
    is_verified = True
    if len(action_array) == 4 :
        return  is_verified, [["verified_model"]+action_array+["Unknown","Unknown"]]
    else:
        return not is_verified, action_array
# 3458 3807 3829 3871
# 3756 3757 3764 3831 3837 3857 3864 3879
def load_data_mnist():
    mnist = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    return mnist, train_data, train_labels, eval_data, eval_labels

def implement_cnn(model_index,flag_verified_model):
    print("model_index ++++++++++++++++++++++++++++++",model_index)
    if not flag_verified_model:
        return tf.estimator.Estimator(model_fn = cnn_model_fn_2)

        return tf.estimator.Estimator(model_fn = cnn_model_fn_2, model_dir = \
            "/vol/bitbucket/nj2217/PROJECT_1/mnist_convnet_model"+ "_"+str(model_index))
        # return tf.estimator.Estimator(model_fn = cnn_model_fn_2model_dir=    \
            # "/vol/bitbucket/nj2217/PROJECT_1/mnist_convnet_model")
    else:
        return tf.estimator.Estimator(model_fn = cnn_model_fn_2)

def set_up_logging():
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)
  return logging_hook

def train_the_model(mnist_classifier,train_data,train_labels,logging_hook):
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": train_data},
                    y=train_labels,
                    batch_size=100,
                    num_epochs=None,
                    shuffle=True)

    mnist_classifier.train(
                    input_fn=train_input_fn,
                    # steps=10000,
                    steps=200,
                    hooks=[logging_hook])

def evaluate_model(mnist_classifier,eval_data,eval_labels):
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)

  return mnist_classifier.evaluate(input_fn=eval_input_fn)

def get_new_model(lastest_model):
    temp_new_model = (lastest_model.strip('model_'))
    new_number = int(temp_new_model)+1
    new_model = "model_"+ str(new_number)

    global model_index
    model_index = new_number
    print("MMMMMOOOOODDDDEEEEELLLL INDEX: ",model_index)
    return new_model

def get_latest_model(action_array):
    file_name = "fixed_model_dict.csv"
    from QLEARNING import open_file

    data = open_file(file_name)
    data = format_data_without_header(data)
    lastest_model = data[-1][0]
    new_model = get_new_model(lastest_model)
    print("new_model : ",new_model)
    new_action_array = [new_model]+action_array+["Unknown","Unknown"]
    return new_action_array

def train(data,action_array):
  flag_verified_data = True
  flag_verified_data = False

  print("train action_array: ",action_array)
  # action_array = get_latest_model(action_array)

  # still some bugs
  # flag_verified_data,action_array = check_format(action_array)
  print("________________ action_array: ",action_array)
  global data1
  global model_index
  temp_action_array = action_array[:]
  data1.append(temp_action_array)

  mnist,train_data,train_labels,eval_data,eval_labels= load_data_mnist()
  mnist_classifier = implement_cnn(model_index,flag_verified_data)
  logging_hook = set_up_logging()
  train_the_model(mnist_classifier,train_data,train_labels,logging_hook)
  eval_results = evaluate_model(mnist_classifier,eval_data,eval_labels)
  print(eval_results)

  file_name = "fixed_model_dict.csv"
  if not flag_verified_data:
      save_to_file(file_name,eval_results)
  data1 = []
  model_index += 1
  print(data1)
  return eval_results['accuracy']

def get_data_from_csv(file_name):
    list_data = []
    with open(file_name, 'rt',encoding='utf8') as f:
        reader = csv.reader(f)
        for row in reader:
            list_data.append(row)
    return list_data[1:]

def pre_train_model(file_name):

    data = get_data_from_csv(file_name)
    # data = [['c_1','c_2','c_3','-']]
    for index in range(len(data)):
    # for index in range(2):
        train(data,data[index])

# def to_verify_model(best_topology):
