"""
author: THomas Athey
date: 7/31/18

Preprocessing for pathology benchmark

Reference:
Color Transfer between IMages
Erik Reinhard, Michael Ashikhmin, Bruce Gooch, Peter Shirley
Applied Perception Sept/Oct 2001

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from stds_from_txt import stds

import tensorflow as tf
import math

slim = tf.contrib.slim

def _color_normalize(image, stds_t):
  """Color normalize via Reinhard method
  Args:
    image: a tensor of size [height, width, 3].
    stds: standard deviations of the lab channels of the training set (found from stds_from_txt.py)
  Returns:
    the color normalized image
  """
  #reshape
  image = tf.transpose(image,perm=[2,0,1])
  image = tf.reshape(image, [3,-1])
  
  #convert to lab
  image = _rgb2lms(image)
  c = tf.constant([10.])
  c = tf.log(c)
  image = tf.div(tf.log(image),c)
  image = _lms2lab(image)
  
  #modify lab values
  means,var = tf.nn.moments(image, axes=[1])
  means = tf.expand_dims(means,1)
  var = tf.expand_dims(var,1)

  image = tf.subtract(image,means)
  stds_i = tf.sqrt(var)


  coeffs = tf.div(stds_t,stds_i)
  image = tf.multiply(image, coeffs)
  image = tf.add(image,means)
  
  #convert back to rgb
  image = _lab2lms(image)
  image = tf.exp(tf.multiply(image,c))
  image = _lms2rgb(image)

  #reshape
  image = tf.reshape(image, [3,1536,2048])
  image = tf.transpose(image, [1,2,0])
  return image
  

def _rgb2lms(image):
  """Converts rgb to lms colors
  Args: 2d tensor of shape [3,num_pixels]
  Returns: 2d tensor of shape [3,num_pixels]
  """

  mat = tf.constant([[0.3811, 0.5783, 0.0402],
                  [0.1967, 0.7244, 0.0782],
                  [0.0241, 0.1288, 0.8444]])
  return tf.matmul(mat,image)

def _lms2lab(image):
  """converts lms to lab colors
  Args: 2d tensor of shape [3, num_pixels]
  Returns: tensor same shape
  """
  mat1 = tf.constant([[math.sqrt(1./3.),0,0],
                      [0,math.sqrt(1./6.),0],
                      [0,0,math.sqrt(1./2.)]])
  mat2 = tf.constant([[1.0,1.0,1.0],[1.0,1.0,-2.0],[1.0,-1.0,0]])
  mat = tf.matmul(mat1,mat2)

  return tf.matmul(mat,image)

def _lab2lms(image):
  """converts lab to lms colors
  Args: 2d tensor of shape [3, num_pixels]
  Returns: tensor same shape
  """
  mat2 = tf.constant([[math.sqrt(3.)/3.,0,0],
                      [0,math.sqrt(6.)/6.,0],
                      [0,0,math.sqrt(2.)/2.]])
  mat1 = tf.constant([[1.0,1.0,1.0],[1.0,1.0,-1.0],[1.0,-2.0,0]])
  mat = tf.matmul(mat1,mat2)

  return tf.matmul(mat,image)


def _lms2rgb(image):
  """Converts lms to rgb colors
  Args: 2d tensor of shape [3,num_pixels]
  Returns: 2d tensor of shape [3,num_pixels]
  """

  mat = tf.constant([[4.4679, -3.5873, 0.1193],
                  [-1.2186, 2.3809, -0.1624],
                  [0.0497, -0.2439, 1.2045]])
  return tf.matmul(mat,image)


def resize(image):
  paddings = tf.stack([[0,32],[0,192],[0,0]]) #hardvoded
  return tf.pad(image,paddings, "CONSTANT")

def preprocess_for_train(image, output_height, output_width, stds_t):
  image.set_shape([1536,2048,3])
  image = tf.to_float(image)
  image = _color_normalize(image,stds_t)
  #image = resize(image)
  image.set_shape([output_height, output_width, 3])
  return image

def preprocess_for_eval(image, output_height, output_width, stds_t):
  image.set_shape([1536,2048,3]) #hardcoded
  image = tf.to_float(image)
  image = _color_normalize(image, stds_t)
  #image = resize(image)
  image.set_shape([output_height, output_width,3])
  return image

def preprocess_image(image, output_height, output_width,txt_file, is_training=False):
  train_stds = stds(txt_file)
  stds_t = tf.constant(train_stds, shape=[3,1], dtype='float32')
  if is_training:
    return preprocess_for_train(image, output_height, output_width, stds_t)
  else:
    return preprocess_for_eval(image, output_height, output_width, stds_t)

