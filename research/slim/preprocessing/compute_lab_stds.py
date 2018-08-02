"""
This script calculates the standard deviation of the lab values which will 
be useful in preprocessing/reinhard_preprocessing.py

Reference:
Color Transfer between IMages
Erik Reinhard, Michael Ashikhmin, Bruce Gooch, Peter Shirley
Applied Perception Sept/Oct 2001

author: Thomas Athey
date: 8/2/18
"""

import tensorflow as tf
import math

def compute_stds(tfrecord):
  """
  computes stds of lab values

  Args: string of path to tfrecord file
  Returns: tensor that contains the 3 stds
  """
  array = _read_data(tfrecord)
  with tf.Session() as sess:
    sess.run(array)
    print(array.shape)

def _read_data(tfrecord):
  """
  turns tfrecord into 4d tensor of images

  Reference:
  http://machinelearninguru.com/deep_learning/data_preparation/tfrecord/tfrecord.html  

  Args: string of path to tfrecord file
  Returns: a 4d tensor of shape [num_images, height, width, 3]
  """

  images = []
  feature={'image/encoded' : tf.FixedLenFeature([], tf.string)}

  filename_queue = tf.train.string_input_producer([tfrecord],num_epochs=1)
  reader = tf.TFRecordReader()
  while True:
    try:
      _, serialized_example = reader.read(filename_queue)
      features = tf.parse_single_example(serialized_example, features=feature)
      image = tf.decode_raw(features['image/encoded'], tf.float32)
      image = tf.reshape(image, [1536, 2048,3])
      image = tf.expand_dims(image,0)
      images = images.append(image
    except OutOfRange:
      break
  return tf.concat(images,0)
