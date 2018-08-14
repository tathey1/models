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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 5, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
 'dataset_name', 'pathology',
  'The name of the dataset with which to compute stats')

tf.app.flags.DEFINE_string(
  'dataset_dir', None,
  'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_string(
  'dataset_split_name', 'train', 'The name of the train/validation/test split')

FLAGS = tf.app.flags.FLAGS

def rgb2lms(pixels):
  """
  converts rgb values to lms values
  
  Args: 2d tensor of shape [3, num_pixels]
  Returns: 2d tensor of shape [3, num_pixels]
  """
  mat = tf.constant([[0.3811, 0.5783, 0.0402],
                     [0.1967, 0.7244, 0.0782],
                     [0.0241, 0.1288, 0.8444]])
  return tf.matmul(mat,pixels)

def lms2lab(pixels):
  """
  converts lms values to lab values
  Args: 2d tensor of shape [3, num_pixels]
  Returns: 2d tensor of shape [3, num_pixels]
  """
  mat1 = tf.constant([[math.sqrt(1/3),0,0],
                      [0,math.sqrt(1/6),0],
                      [0,0,math.sqrt(1/2)]])
  mat2 = tf.constant([[1.0,1.0,1.0],[1.0,1.0,-2.0],[1.0,-1.0,0]])
  mat = tf.matmul(mat1,mat2)
  return tf.matmul(mat,pixels)

def compute_stds(images):
  """
  Computes standard deviations in lab space

  Args: 4d tensor of shape [num_images, height, width, 3]
  Returns: tensor of length 3
  """
  images_reshaped = tf.transpose(images,perm=[3,0,1,2])
  images_reshaped = tf.reshape(images_reshaped,[3,-1])
  LMS = rgb2lms(images_reshaped)
  log_LMS = tf.log(LMS)
  lab = lms2lab(log_LMS)
  _, var = tf.nn.moments(lab, axes=1)
  return tf.sqrt(var)

def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  with tf.Graph().as_default():
    dataset = dataset_factory.get_dataset(
      FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    provider = slim.dataset_data_provider.DatasetDataProvider(dataset,
      shuffle=False, common_queue_capacity = 2*FLAGS.batch_size,
      common_queue_min=FLAGS.batch_size)
    [image, label] = provider.get(['image','label'])
    image.set_shape([1536, 2048, 3])
    images, labels = tf.train.batch([image,label], batch_size=FLAGS.batch_size,
                              num_threads=FLAGS.num_preprocessing_threads,
                              capacity=5*FLAGS.batch_size)
    
    print(images)
    stds = compute_stds(tf.cast(images, dtype=tf.float32))
    with tf.Session() as sess:
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(sess=sess, coord=coord)
      print(stds.eval())

if __name__ == '__main__':
  tf.app.run()
