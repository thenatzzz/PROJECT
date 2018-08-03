import numpy as np
import tensorflow as tf
tf.reset_default_graph()

# Create some variables.
# v1 = tf.get_variable("v1", shape=[3])
# v2 = tf.get_variable("v2", shape=[5])

# Add ops to save and restore all the variables.
# saver = tf.train.Saver()

'''
# Later, launch the model, use the saver to restore variables from disk, and
# do some work with the model.
with tf.Session() as sess:
  # Restore variables from disk.
  path_to_model = "/homes/nj2217/PROJECT/tensorflow_tut/MODEL_TO_TRAIN/model_1/mnist_convnet_model"
  model_file = "model.ckpt-2001.meta"
  saver.restore(sess, path_to_model+"/"+model_file)

  # saver.restore(sess, "/tmp/model.ckpt")
  print("Model restored.")
  # Check the values of the variables
  print("v1 : %s" % v1.eval())
  print("v2 : %s" % v2.eval())
'''

with tf.Session() as sess:
    path_to_model = "/homes/nj2217/PROJECT/tensorflow_tut/MODEL_TO_TRAIN/model_1/mnist_convnet_model"
    meta_model_file = "model.ckpt-1.meta"
    ckpt_model_file = "model.ckpt-1"
    saver = tf.train.import_meta_graph(path_to_model+"/"+meta_model_file)
    print(saver.__dict__, " ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    saver.restore(sess, path_to_model+"/"+ckpt_model_file)
    print('\n\n')
    print(saver.__dict__, " :::::::::::::::::::::::::::::    ")
