"""
This script calculates the standard deviation of the lab values which will 
be useful in preprocessing/reinhard_preprocessing.py
Reference:
Color Transfer between IMages
Erik Reinhard, Michael Ashikhmin, Bruce Gooch, Peter Shirley
Applied Perception Sept/Oct 2001
author: Thomas Athey
date: 8/3/18
"""

import math
import numpy as np
from scipy import ndimage

def stds(filename):
  files = _get_filenames(filename)
  array = _read_images(files)
  stds = _calculate_stds(array)
  
  return stds

def _get_filenames(filename):
  f = open(filename, 'r')
  text = f.read()
  end = text.find('Validation Files:')
  text = text[:end]
  files = text.splitlines()
  files = files[1:]
  return files

def _read_images(files):
  
  for i in range(len(files)):
    print(i)
    im = ndimage.imread(files[i])
    im = np.expand_dims(im,axis=0)
    if i==0:
      ims=im
    else:
      ims = np.concatenate((ims,im),axis=0)

  return ims

def _calculate_stds(array):
  """
  Calculates the standard deviations

  Args: 4d array of shape [num_images,height, width, 3]
  Returns: array of length 3
  """
  print('Reshaping...')
  rgb = np.moveaxis(array, 3, 0)
  rgb = np.reshape(rgb, (3,-1))
  print('RGB')
  print(rgb[:,0])
  print('Converting to lab...')
  lms = rgb2lms(rgb)
  del rgb
  print('LMS')
  print(lms[:,0])
  log_lms = np.log(lms)
  del lms
  print('Log LMS')
  print(log_lms[:,0])
  lab = lms2lab(log_lms)
  del log_lms
  print('LAB')
  print(lab[0,:])
  print('Computing Standard deviations...')
  return np.std(lab, axis=1)

def rgb2lms(rgb):
  """
  Converts rgb to lms

  Args: 2d array shape [3, num_pixels]
  Returns: 2d array shape [3, num_pixels]
  """
  mat = np.array([[0.3811, 0.5783, 0.0402],
                  [0.1967, 0.7244, 0.0782],
                  [0.0241, 0.1288, 0.8444]])

  return np.matmul(mat, rgb)

def lms2lab(lms):
  """
  Converts lms to lab

  Args: 2d array shape [3. num_pixels]
  Returns: 2d array shape [3, num_pixels]
  """

  mat1 = np.array([[math.sqrt(1./3.),0,0],
                      [0,math.sqrt(1./6.),0],
                      [0,0,math.sqrt(1./2.)]])
  mat2 = np.array([[1.0,1.0,1.0],[1.0,1.0,-2.0],[1.0,-1.0,0]])
  mat = np.matmul(mat1,mat2)

  return np.matmul(mat,lms)

