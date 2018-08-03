from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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

# NAME OF FILE TO OPEN (csv format)
file_csv = "model_dict.csv"
data1 = []
global_index = 0
model_index = 0

# with open(file_csv) as f:
#     reader = csv.reader(f, delimiter=',' , quotechar= ',' ,
#                         quoting = csv.QUOTE_MINIMAL)
#     data1  = [r for r in reader]
#     data1 = data1[1:]      # To get rid of column name
# model_index = int(data1[-1][0].replace("model_",""))
# true_index = len(data1)
# global_index= true_index-1


# file_name_to_update = file_csv
# if os.path.isfile(file_name_to_update):
#     print("inside isfile function")
#     number_of_row = sum(1 for line in open(file_name_to_update))
#     global_index = number_of_row-1
# else:
#     global_index  = 0

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
                                    strides=num_stride)

def cnn_model_fn_2(features,labels, mode):
    input_layer = tf.reshape(features["x"], [-1,28,28,1])
    # print("|||||||||||||||||GLOBAL index: ",global_index)
    # print("length data1: ",len(data1))
    # print(data1[global_index])
    # print("__________ " ,data1[global_index][:-2])

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

def get_latest_model():
    file_path = "/homes/nj2217/PROJECT/REINFORCEMENT_algo/"
    file_name = "model_dict.csv"
    file_csv = file_path + file_name
    with open(file_csv) as f:
        reader = csv.reader(f, delimiter=',' , quotechar= ',' ,
                        quoting = csv.QUOTE_MINIMAL)
        data  = [r for r in reader]
        print("Length data inside get_latest_model: ", len(data))
    return str(data[-1][0])

def get_new_model(lastest_model):
    temp_new_model = (lastest_model.strip('model_'))
    new_number = int(temp_new_model)+1
    new_model = "model_"+ str(new_number)

    global model_index
    model_index = new_number
    return new_model

def format_action_array(action_array):
    lastest_model = get_latest_model()
    new_model = get_new_model(lastest_model)
    unknown_col = ["Unknown","Unknown"]
    formatted_list =  [new_model] + action_array + unknown_col
    return formatted_list

def train(action_array):

  global data1
  temp_action_array = action_array[:]
  print("action_array inside train file: ",action_array)
  print("temp_action_array inside train file: ",temp_action_array)

  temp_data1 = format_action_array(temp_action_array)
  unused_list = ["unused"]
  data1.append(temp_data1)
  data1.append(unused_list)
  print("data1: ", data1)
  print("model_index: ", model_index)
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images  # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images  # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  csv_columns = ['Model', '1st Layer', '2nd Layer','3rd Layer', '4th Layer','Accuracy', 'Loss']
  list_of_dict = []
  list_of_accuracy = []

  list_of_dict = []
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn_2, model_dir=    \
      "/vol/bitbucket/nj2217/PROJECT_3/mnist_convnet_model"+ \
      "_"+str(model_index))


  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

      # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      # steps=20000,
      steps=200,
      hooks=[logging_hook])

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  print('\n\n')
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)
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

  path_to_file = "/homes/nj2217/PROJECT/REINFORCEMENT_algo/"
  csv_file = path_to_file+"model_dict.csv"

  try:
     with open(csv_file, 'a') as csvfile:

         writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
         if os.stat(csv_file).st_size == 0 : # ONLY write ROW Hedaer when file is empty
            writer.writeheader()
         for data in list_of_dict:
             writer.writerow(data)
  except IOError:
     print("I/O error")

  data1 = []
  return temp_dict['Accuracy']
# train()
