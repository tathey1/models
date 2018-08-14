"""
author: Thomas Athey tathey1@jhmi.edu
date: 7/27/18
Code plaigarized from tensorflow/models/research/slim/datasets/download_and_convert_flowers.py
See this^ file for comments
Converts pathology data to TFRecords of TF-Example protos
This module reads the pathology pictures that make up the pathology data
and creates two TFRecord datasets: one for train and one for test.
Each TFRecord dataset is comprised of a set of TF-Example protocol buffers, 
each of which conain a single image and label
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

from datasets import dataset_utils
from datasets.download_and_convert_flowers import ImageReader

_NUM_VALIDATION = 40
_NUM_TRAIN = 360
_RANDOM_SEED = 0
_NUM_SHARDS = 1

def _get_filenames_and_classes(dataset_dir):
	"""Returns a list of filenames and inferred class names
	Args: dataset_dir a directory with subdirectories representing class names
	Each subdirectory should contain png or jpg encoded images.
	Returns: a list of image file paths, relative to 'dataset_dir' and the
	list of subdirectories, representing class names
	"""
	directories = []
	class_names = []
	for filename in os.listdir(dataset_dir):
		path = os.path.join(dataset_dir,filename)
		if os.path.isdir(path):
			directories.append(path)
			class_names.append(filename)

	photo_filenames = []
	for directory in directories:
		for filename in os.listdir(directory):
			path = os.path.join(directory, filename)
			photo_filenames.append(path)

	return photo_filenames, sorted(class_names)

def _get_dataset_filename(dataset_dir, split_name, shard_id):
	output_filename = 'pathology_%s_%05d-of-%05d.tfrecord' % (
		split_name, shard_id, _NUM_SHARDS)
	return os.path.join(dataset_dir, output_filename)

def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
	"""Converts the given filenames to a TFRecord dataset.
	Args:
	split_name: the name of the dataset, either 'train' or 'validation'.
	filenames: a list of absolute paths to png or jpg images
	class_names_to_ids: A dictionary from class names (strings)
	to ids (integers).
	dataset_dir: the directory where the converted datasets are stored.
	"""
	assert 'train' in split_name or 'validation' in split_name#,'test']

	num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))
	
	g = tf.Graph()
	with g.as_default():
		image_reader = ImageReader()

		with tf.Session('') as sess:
			for shard_id in range(_NUM_SHARDS):
				output_filename = _get_dataset_filename(
					dataset_dir, split_name, shard_id)
				with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
					start_ndx = shard_id * num_per_shard
					end_ndx = min((shard_id+1)*num_per_shard, len(filenames))
					for i in range(start_ndx, end_ndx):
						sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
							i+1, len(filenames), shard_id))
						sys.stdout.flush()

						image_data = tf.gfile.FastGFile(filenames[i],'rb').read()
						height, width = image_reader.read_image_dims(sess, image_data)
						class_name = os.path.basename(os.path.dirname(filenames[i]))
						class_id = class_names_to_ids[class_name]
						
						example = dataset_utils.image_to_tfexample(
							image_data, b'jpg', height, width, class_id)
						tfrecord_writer.write(example.SerializeToString())
	sys.stdout.write('\n')
	sys.stdout.flush()

def _dataset_exists(dataset_dir):
	for split_name in ['train','validation']:
		for shard_id in range(_NUM_SHARDS):
			output_filename = _get_dataset_filename(
				dataset_dir, split_name, shard_id)
			if not tf.gfile.Exists(output_filename):
				return False
	return True

def run(dataset_dir):
	"""Runs the conversion operation
	Args:
	dataset_dir: the dataset directory where the dataset is stored
	"""

	photo_filenames, class_names = _get_filenames_and_classes(dataset_dir)
	class_names_to_ids = dict(zip(class_names, range(len(class_names))))

	random.seed(_RANDOM_SEED)
	random.shuffle(photo_filenames)
	for split_num in range(10):
		start_idx=split_num*_NUM_VALIDATION
		end_idx=(_NUM_TRAIN+start_idx)
		idxs=range(start_idx,end_idx)
		l=len(photo_filenames)
		training_filenames = [photo_filenames[i % l] for i in idxs]

		idxs=range(end_idx,end_idx+_NUM_VALIDATION)
		validation_filenames = [photo_filenames[i % l] for i in idxs]

		fname='pathology_splits_' + str(split_num) + '.txt'        
	        path = os.path.join(dataset_dir,fname)
	        print('Creating file: ' + path)
	        f = open(path,'w+')
        	
	        f.write('Training Files:\n')
	        for training_filename in training_filenames:
	          f.write(training_filename + '\n')
	        
	        f.write('Validation Files:\n')
	        for validation_filename in validation_filenames:
	          f.write(validation_filename + '\n')
	        #f.write('Testing Files:\n')
	        #for testing_filename in testing_filenames:
	        #  f.write(testing_filename + '\n')

		_convert_dataset('train_'+str(split_num), training_filenames, class_names_to_ids, dataset_dir)
		_convert_dataset('validation_'+str(split_num), validation_filenames, class_names_to_ids, dataset_dir)
	#	_convert_dataset('test', testing_filenames, class_names_to_ids, dataset_dir)
	
		#labels_to_class_names = dict(zip(range(len(class_names)), class_names))
		#dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

	print('\nFinished converting the Pathology dataset!')
