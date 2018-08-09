"""
author: Thomas Athey
date: 7/30/18

Code heavily borrowed from tensorflow/models/research/slim/datasets/flowers.py

Provides data for the pathology dataset
"""

from __future__ import absolute_import, division, print_function

import os
import tensorflow as tf

from datasets import dataset_utils

slim = tf.contrib.slim

_FILE_PATTERN = 'pathology_%s_*.tfrecord'

SPLITS_TO_SIZES = {'train': 360, 'validation': 40}

_NUM_CLASSES = 4

_ITEMS_TO_DESCRIPTIONS = {
	'image': 'A color image of varying size.',
	'label': 'A single integer between 0 and 3',
}

def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
	"""Gets a dataset tuple with instructions for reading pathology images.

	Args: split_name: a train/validation/test split name
	dataset_dir: the base directory of the dataset sources
	file_pattern: the file pattern to use when matching the dataset sources.
		It is assumed that the pattern contains a '%s' string so that the
		split name can be inserted
	reader: a TensorFlow reader type

	Returns:
	A 'Dataset' namedtuple

	Raises:
	ValueError: if 'split_name' is not a valid train/validation/test split.
	"""
	if split_name not in SPLITS_TO_SIZES and split_name[:-2] not in SPLITS_TO_SIZES:
		raise ValueError('split name %s was not recognized.' % split_name)

	if not file_pattern:
		file_pattern = _FILE_PATTERN
	file_pattern = os.path.join(dataset_dir, file_pattern % split_name)
	print('******************************************************')
	print(file_pattern)
	#Allowing None in the signature so that the dataset_factory can use the default
	if reader is None:
		reader = tf.TFRecordReader

	keys_to_features = {
		'image/encoded': tf.FixedLenFeature(
			(), tf.string, default_value=''),
		'image/format': tf.FixedLenFeature(
			(), tf.string, default_value='jpeg'),
		'image/class/label': tf.FixedLenFeature(
			[], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
	}
	items_to_handlers = {
		'image': slim.tfexample_decoder.Image(),
		'label': slim.tfexample_decoder.Tensor('image/class/label'),
	}

	decoder = slim.tfexample_decoder.TFExampleDecoder(
		keys_to_features, items_to_handlers)

	labels_to_names = None
	if dataset_utils.has_labels(dataset_dir):
		labels_to_names=dataset_utils.read_label_file(dataset_dir)
	
	if split_name in SPLITS_TO_SIZES:
		num_samples = SPLITS_TO_SIZES[split_name]
	elif split_name[:-2] in SPLITS_TO_SIZES:
		num_samples = SPLITS_TO_SIZES[split_name[:-2]]

	return slim.dataset.Dataset(data_sources=file_pattern,
		reader=reader,
		decoder=decoder,
		num_samples=num_samples,
		items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
		num_classes=_NUM_CLASSES,
		labels_to_names=labels_to_names)
