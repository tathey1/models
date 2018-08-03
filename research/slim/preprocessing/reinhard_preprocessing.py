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

import tensorflow as tf

slim = tf.contrib.slim

_l_std = 0.45838048
_a_std = 0.15733315
_b_std = 0.01821799

def _color_normalize(image, stds):
  """Subtracts the given means from each image channel.
  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)
  Note that the rank of `image` must be known.
  Args:
    image: a tensor of size [height, width, 3].
    stds: standard deviations of the lab channels of the training set (found from stds_from_txt.py)
  Returns:
    the color normalized image
  """
  

def resize(image):
  paddings = tf.stack([[0,32],[0,192],[0,0]]) #hardvoded
  return tf.pad(image,paddings, "CONSTANT")

def preprocess_for_train(image, output_height, output_width):
  image = tf.to_float(image)
  image = _color_normalize(image,[_l_std, _a_std, _b_std])
  image = resize(image)
  image.set_shape([output_height, output_width, 3])
  return image

def preprocess_for_eval(image, output_height, output_width):
  image = tf.to_float(image)
  image = _color_normalize(image, [_l_std, _a_std, _b_std])
  image = resize(image)
  image.set_shape([output_height, output_width,3])
  image = tf.to_float(image)
  return image

def preprocess_image(image, output_height, output_width, is_training=False):
  if is_training:
    return preprocess_for_train(image, output_height, output_width)
  else:
    return preprocess_for_eval(image, output_height, output_width)

