from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from layer_algo import fn_to_add_column

import numpy as np
import tensorflow as tf
import csv
import os
import pandas as pd
# import os.path

tf.logging.set_verbosity(tf.logging.INFO)
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
m_1 = [2,2]
m_2 = [3,2]
m_3 = [5,3]
s = [0]

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

data1 = create_model()
print(data1)

global_index = 0
print("data1[global_index][-2]: ",data1[global_index][-2])

def count_model_layer(model):
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
    print("NUM LAYER: ", num_layer, " @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

    layer = input_layer
    temp_layer = 0
    for index  in range(1,num_layer):
        print("layer trained: ", data1[global_index][index])
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

def main(unused_argv):
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images  # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images  # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  global global_index
  csv_columns = ['Model', '1st Layer', '2nd Layer','3rd Layer', '4th Layer','Accuracy', 'Loss']
  list_of_dict = []
  list_of_accuracy = []

  how_many_model_to_train = 100
  temp_global_index = global_index
  while global_index < temp_global_index+how_many_model_to_train:
  # while global_index < 20:
      list_of_dict = []
      mnist_classifier = tf.estimator.Estimator(
          model_fn=cnn_model_fn_2, model_dir=    \
          "/vol/bitbucket/nj2217/"+ \
          "_"+str(global_index))
      tensors_to_log = {"probabilities": "softmax_tensor"}
      logging_hook = tf.train.LoggingTensorHook(
          tensors=tensors_to_log, every_n_iter=50)

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
      print('\n')
      print("EVAL INPUT FN: ",eval_input_fn)
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

      global_index += 1

if __name__ == "__main__":

  tf.app.run()
