"""
author: Thomas Athey
date: 7/30/18

Significant code is borrowed from
tensorflow/models/research/slim/datasets/download_and_convert+flowers.py

Convert a particular directory of tiff files to jpegs.
The directory should have subdirectories corresponding to the different classes. 
Jpegs will be found in a directory at the same level of the input
directory, with the same name + "tiff" appended at the end

Usage:
$ python convert_tiff_to_jpeg.py \
	--dir=/workspace/data/pathology

"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import os
from PIL import Image


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
	'directory',
	None,
	'The directory that contains .tiff images')

def _get_filenames_and_classes(dataset_dir):
	"""Returns a list of filenames and inferred class names
	Args: dataset_dir a directory with subdirectories representing class names
	Each subdirectory should contain tiff images

	Returns: the name of a corresponding jpeg parent directory
	a list of tiff subdirectories
	and the enclosed image tiff files paths, relative to 'dataset_dir' 
	Also a corresponding list of jpeg subdirectories and file paths
	that should be produced
	"""
	
	jpeg_parent = dataset_dir[:-1] + "jpeg"

	jpeg_directories = []
	tiff_directories = []

	for filename in os.listdir(dataset_dir):
		path = os.path.join(dataset_dir,filename)
		jpeg_path = os.path.join(jpeg_parent,filename)
		if os.path.isdir(path):
			tiff_directories.append(path)
			jpeg_directories.append(jpeg_path)
	

	tiff_filenames = []
	jpeg_filenames = []

	for i in range(len(tiff_directories)):
		tiff_directory = tiff_directories[i]
		jpeg_directory = jpeg_directories[i]
		for filename in os.listdir(tiff_directory):
			path = os.path.join(tiff_directory, filename)
			tiff_filenames.append(path)

			jpeg_filename = filename[:-3] + "jpeg"
			jpeg_path = os.path.join(jpeg_directory,jpeg_filename)
			jpeg_filenames.append(jpeg_path)

	return tiff_directories, jpeg_directories, tiff_filenames, jpeg_filenames

def _make_jpeg_dirs(jpeg_dirs):
	for directory in jpeg_dirs:
		try:
			if not os.path.exists(directory):
				os.makedirs(directory)
			else:
				print('Directory: ' + directory + ', already exists')
		except:
			print('Error: Creating directory: ' + directory)

def _convert_tiff(tiffs, jpegs):
	for i in range(len(tiffs)):
		tiff = tiffs[i]
		jpeg = jpegs[i]

		if not os.path.exists(jpeg):
			im = Image.open(tiff)
			print('Generating jpeg for %s' % tiff)
			im.save(jpeg)
		else:
			print('File: ' + jpeg + ', already exists')

def main(_):
	if not FLAGS.directory:
		raise ValueError('You must supply the directory with --directory')


	dataset_dir = FLAGS.directory

	tiff_dir, jpeg_dir, tiff_files, jpeg_files = _get_filenames_and_classes(dataset_dir)
	
	_make_jpeg_dirs(jpeg_dir)
	_convert_tiff(tiff_files, jpeg_files)

if __name__ == '__main__':
	tf.app.run()
