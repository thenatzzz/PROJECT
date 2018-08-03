from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

def get_conv(input_layer,num_filter, kernel_size):
    return tf.layers.conv2d(
        inputs=input_layer,
        filters= num_filter,
        kernel_size=[kernel_size, kernel_size],
        padding="same",
        activation=tf.nn.relu)
def get_pool(input_layer, size_kernel, num_stride):
    return tf.layers.max_pooling2d(inputs=input_layer, pool_size=[size_kernel,size_kernel],strides=num_stride)

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer
  # Reshape X to 4-D tensor: [batch_size, width, height, channels]
  # MNIST images are 28x28 pixels, and have one color channel

  '''c_10 = [64,3,1], m_1 = [2,2], c_6=[36,5,1],c_10 = [64,3,1] '''
  # input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
  # layer_1 = get_conv(input_layer,28,3)
  # layer_2 = get_pool(layer_1,2,2)
  # layer_3 = get_conv(layer_2,36,5)
  # layer_4 = get_conv(layer_3,64,3)
  # last_layer = layer_4

  '''  c_8=[48,4,1], c_12=[64,5,1],c_12=[64,5,1],c_4=[36,3,1] '''
  # input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
  # layer_1 = get_conv(input_layer,48,1)
  # layer_2 = get_conv(layer_1,64,5)
  # layer_3 = get_conv(layer_2,64,5)
  # layer_4 = get_conv(layer_3,36,3)
  # last_layer = layer_4

  '''  c_11=[64,4,1], c_10 = [64,3,1], m_2=[3,2] '''
  # input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
  # layer_1 = get_conv(input_layer,64,4)
  # layer_2 = get_conv(layer_1,64,3)
  # layer_3 = get_pool(layer_2,3,2)
  # last_layer = layer_3

  ''' model_437: m_3=[5,3],m_2=[3,2],m_3=[5,3],c_8=[48,4,1] '''
  # input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
  # layer_1 = get_pool(input_layer,5,3)
  # print("layer_1 shape: ", layer_1.get_shape())
  # layer_2 = get_pool(layer_1,3,2)
  # print("layer_2 shape: ", layer_2.get_shape())
  # layer_3 = get_pool(layer_2,5,3)
  # print("layer_3 shape: ", layer_3.get_shape())
  # layer_4 = get_conv(layer_3,48,4)
  # print("layer_4 shape: ", layer_4.get_shape())
  # last_layer = layer_4

  '''  m_3=[5,3],m_2=[3,2],c_8=[48,4,1],c_8=[48,4,1] '''
  input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
  layer_1 = get_pool(input_layer,5,3)
  layer_2 = get_pool(layer_1,3,2)
  layer_3 = get_conv(layer_2,48,4)
  layer_4 = get_conv(layer_3,48,4)
  last_layer = layer_4



  '''
  # c_10 = [64,3,1], m_1 = [2,2], c_6=[36,5,1],c_10 = [64,3,1]
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=36,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  conv3 = tf.layers.conv2d(
      inputs=conv2,
      filters=64,
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)
  last_layer = conv3
  '''
  shape_array = last_layer.get_shape()
  # Flatten tensor into a batch of vectors
  # Input Tensor Shape: [batch_size, 7, 7, 64]
  # Output Tensor Shape: [batch_size, 7 * 7 * 64]
  pool2_flat = tf.reshape(last_layer, [-1, shape_array[1]*shape_array[2]*shape_array[3]])

  # Dense Layer
  # Densely connected layer with 1024 neurons
  # Input Tensor Shape: [batch_size, 7 * 7 * 64]
  # Output Tensor Shape: [batch_size, 1024]
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

  # Add dropout operation; 0.6 probability that element will be kept
  dropout = tf.layers.dropout(
      inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)

  # Logits layer
  # Input Tensor Shape: [batch_size, 1024]
  # Output Tensor Shape: [batch_size, 10]
  logits = tf.layers.dense(inputs=dropout, units=10)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
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

  # Create the Estimator
  path_saved_model ="/homes/nj2217/PROJECT/tensorflow_tut/MODEL_TO_TRAIN/model_1/mnist_convnet_model"
  mnist_classifier = tf.estimator.Estimator(
      model_fn=cnn_model_fn, model_dir=path_saved_model)

      # model_fn=cnn_model_fn, model_dir="/tmp/nj2217/folder_cnn_mnist/mnist_convnet_model")
      # model_fn=cnn_model_fn, model_dir="/vol/bitbucket/nj2217/folder_cnn_mnist/mnist_convnet_model")

  # Set up logging for predictions
  # Log the values in the "Softmax" tensor with label "probabilities"
  tensors_to_log = {"probabilities": "softmax_tensor"}
  logging_hook = tf.train.LoggingTensorHook(
      tensors=tensors_to_log, every_n_iter=50)

  #Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)
  mnist_classifier.train(
      input_fn=train_input_fn,
      # steps=20000,
      steps=2000,

      hooks=[logging_hook])

  # # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()
