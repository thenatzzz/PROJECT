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
file_csv = "Model1501-2000.csv"
# file_csv = "Names_11_30.csv"
# global_index = 0

'''
with open(file_csv) as f:
    reader = csv.DictReader(f)
    data = [r for r in reader]
'''

with open(file_csv) as f:
    reader = csv.reader(f, delimiter=',' , quotechar= ',' ,
                        quoting = csv.QUOTE_MINIMAL)
    data1  = [r for r in reader]
    data1 = data1[1:]      # To get rid of column name

# print("\n\n ###############################################")
# print("data1[0] : ",data1[0])
# print("data1[0][0] : ", data1[0][0])
model_index = int(data1[0][0].replace("model_",""))
# print("Type of model_index: ", type(model_index))
# print("data1[0][0] : ", model_index)
# print("________________________",data1[1])
# print("data1[global_index]")
# def get_global_index
global_index = 0
# print("data1[global_index][-2]: ",data1[global_index][-2])


# file_name_to_update = "MODEL_2.csv"
file_name_to_update = "MODEL_dict.csv"
if os.path.isfile(file_name_to_update):
# if os.path.isfile("./tensorflow_tut/MODEL_2.csv"):
# if os.path.isfile(file_to_check):
    print("inside isfile function")
    number_of_row = sum(1 for line in open(file_name_to_update))
    global_index = number_of_row-1
    # global_index = number_of_row

    print(number_of_row)
else:
    global_index  = 0
print("\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print("GLOBAL INDEX: ", global_index)
print("TRAINING MODEL: ", global_index+1)
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

# while data1[global_index][-2] != "Unknown":
#     global_index += 1


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
                                    strides=num_stride)

def cnn_model_fn_2(features,labels, mode):
    input_layer = tf.reshape(features["x"], [-1,28,28,1])

    num_layer = count_model_layer(data1[global_index][:-2])
    print("NUM LAYER: ", num_layer, " @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

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
        # print("TEMP LAYER at the last: ", temp_layer)
        layer = temp_layer
        # print("GET LAYER SIZE : ", layer.get_shape())

    shape_array = layer.get_shape()
    # print(shape_array[1],"+",shape_array[2],"+",shape_array[3])
    # print("GET LAYER SIZE : ", layer.get_shape())
    # pool2_flat = tf.reshape(layer, [-1, 7 * 7 * 64])
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

def fn_to_change_content(accuracy,loss):
    '''
    # Open file
    r = csv.reader(open(file_csv))
    lines = list(r)
    # chnage/add Accuracy which is at 2nd last column
    lines[global_index][-2] = accuracy
    # change/add Loss which is at last column
    lines[global_index][-1] = loss

    # Write back to file
    writer = csv.writer(open(file_csv), 'w')
    writer.writerows(lines)
    '''
    # Read from csv file
    df = pd.read_csv(file_csv)
    # Set value in Accuracy/Loss column at specific row(global_index)
    df.set_value(global_index, 'Accuracy', accuracy)
    df.set_value(global_index, 'Loss', loss)
    # Save the changed values to file
    df.to_csv(file_csv, index=False)

def main(unused_argv):
  # Load training and eval data
  mnist = tf.contrib.learn.datasets.load_dataset("mnist")
  train_data = mnist.train.images  # Returns np.array
  train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
  eval_data = mnist.test.images  # Returns np.array
  eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

  global model_index
  global global_index
  # csv_columns = ['Model', '1st Layer', '2nd Layer','3rd Layer', '4th Layer']
  csv_columns = ['Model', '1st Layer', '2nd Layer','3rd Layer', '4th Layer','Accuracy', 'Loss']
  list_of_dict = []
  list_of_accuracy = []

  # Create the Estimator
  # while global_index < len(data1):
  # how_many_model_to_train = 5
  how_many_model_to_train = 500

  temp_global_index = global_index
  while global_index < temp_global_index+how_many_model_to_train:
  # while global_index < 20:
      list_of_dict = []
      #temp_dict = {}
      mnist_classifier = tf.estimator.Estimator(
          # model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model"+"_"+str(global_index+1))

          # model_fn=cnn_model_fn_2, model_dir="/tmp/nj2217/mnist_convnet_model"+"_"+str(global_index+1))

          # model_fn=cnn_model_fn_2, model_dir="/tmp/nj2217_2/mnist_convnet_model"+"_"+str(global_index+1))

          # model_fn=cnn_model_fn_2, model_dir="/vol/bitbucket/nj2217/PROJECT/mnist_convnet_model"+"_"+str(global_index+1))
          model_fn=cnn_model_fn_2, model_dir=    \
          # "/vol/bitbucket/nj2217/PROJECT_1/mnist_convnet_model"+ \
          "/vol/bitbucket/nj2217/PROJECT_2/mnist_convnet_model"+ \
          "_"+str(model_index+global_index))


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
          steps=20000,
          # steps=200,
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
      # output of eval_results = {'accuracy': 0.6019, 'loss': 2.1334257, 'global_step': 200}
      # print("Accuracy: ",eval_results['accuracy'])
      # print("Loss: ",eval_results['loss'])
      # print("Global step: ",eval_results['global_step'])
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

      path_to_file = "/vol/bitbucket/nj2217/PROJECT_2/"
      csv_file = path_to_file+"MODEL_dict_7.csv"

      try:
         # with open(csv_file, 'w') as csvfile:
         with open(csv_file, 'a') as csvfile:

             writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
             if os.stat(csv_file).st_size == 0 : # ONLY write ROW Hedaer when file is empty
                writer.writeheader()
             for data in list_of_dict:
                 writer.writerow(data)
      except IOError:
         print("I/O error")

      global_index += 1

'''
  path_to_file = "/vol/bitbucket/nj2217/PROJECT_2/"
  csv_file = path_to_file+"MODEL_dict_2.csv"

  try:
     # with open(csv_file, 'w') as csvfile:
     with open(csv_file, 'a') as csvfile:

         writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
         if os.stat(csv_file).st_size == 0 : # ONLY write ROW Hedaer when file is empty
            writer.writeheader()
         for data in list_of_dict:
             writer.writerow(data)
  except IOError:
     print("I/O error")
'''
if __name__ == "__main__":
  tf.app.run()
