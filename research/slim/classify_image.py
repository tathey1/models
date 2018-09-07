'''
Takes a model checkpoint and makes predictions on a dataset and writes these results to a file.
With the predictions, we can make ROC curves and confusion matrices etc.

make sure the network name has been added to model_name_to_variables

Example usage:
CUDA_VISIBLE_DEVICES=1 python classify_image.py --num_classes=4 \
--infile=/workspace/data/Part-A_Originaljpeg/pathology_validation_0_00000-of-00001.tfrecord \
--tfrecord=True --outfile=/workspace/results/xval_final_15000/predict/predict.txt \
--model_name=resnet_v1_50_final \
--checkpoint_path=/workspace/results/xval_final_15000/train_logs/val_0/model.ckpt-15000
'''


#!/usr/bin/env python

from __future__ import print_function
import sys
sys.path.append('../../tensorflow/models/slim/') # add slim to PYTHONPATH
import tensorflow as tf

tf.app.flags.DEFINE_integer('num_classes', 5, 'The number of classes.')
tf.app.flags.DEFINE_string('infile',None, 'Image file, one image per line.')
tf.app.flags.DEFINE_boolean('tfrecord',False, 'Input file is formatted as TFRecord.')
tf.app.flags.DEFINE_string('outfile',None, 'Output file for prediction probabilities.')
tf.app.flags.DEFINE_string('model_name', 'resnet_v1_50', 'The name of the architecture to evaluate.')
tf.app.flags.DEFINE_string('preprocessing_name', None, 'The name of the preprocessing to use. If left as `None`, then the model_name flag is used.')
tf.app.flags.DEFINE_string('checkpoint_path', 'finetuned_checkpoints/resnet_v1_50/','The directory where the model was written to or an absolute path to a checkpoint file.')
tf.app.flags.DEFINE_integer('eval_image_size', None, 'Eval image size.')
FLAGS = tf.app.flags.FLAGS

import numpy as np
import os

from datasets import imagenet
from nets import inception
from nets import resnet_v1
from nets import inception_utils
from nets import resnet_utils
from preprocessing import inception_preprocessing
from nets import nets_factory
from preprocessing import preprocessing_factory

slim = tf.contrib.slim

model_name_to_variables = {'resnet_v1_50_final':'resnet_v1_50_final','resnet_v1_50_pathology_benchmark':'resnet_v1_50_pathology_benchmark','inception_v3':'InceptionV3','inception_v4':'InceptionV4','resnet_v1_50':'resnet_v1_50','resnet_v1_152':'resnet_v1_152'}

preprocessing_name = FLAGS.preprocessing_name or FLAGS.model_name
eval_image_size = FLAGS.eval_image_size

if FLAGS.tfrecord:
  fls = tf.python_io.tf_record_iterator(path=FLAGS.infile)
else:
  fls = [s.strip() for s in open(FLAGS.infile)]

model_variables = model_name_to_variables.get(FLAGS.model_name)
if model_variables is None:
  tf.logging.error("Unknown model_name provided `%s`." % FLAGS.model_name)
  sys.exit(-1)

if FLAGS.tfrecord:
  tf.logging.warn('Image name is not available in TFRecord file.')

if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
  checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
else:
  checkpoint_path = FLAGS.checkpoint_path

image_string = tf.placeholder(tf.string) # Entry to the computational graph, e.g. image_string = tf.gfile.FastGFile(image_file).read()

#image = tf.image.decode_image(image_string, channels=3)
image = tf.image.decode_jpeg(image_string, channels=3, try_recover_truncated=True, acceptable_fraction=0.3) ## To process corrupted image files

if FLAGS.model_name == 'resnet_v1_50_pathology_benchmark':
  txt_file = '/workspace/data/Part-A_Originaljpeg/pathology_splits_0.txt'
else:
  txt_file=None

image_preprocessing_fn = preprocessing_factory.get_preprocessing(preprocessing_name,
 txt_file=txt_file, is_training=False)

network_fn = nets_factory.get_network_fn(FLAGS.model_name, FLAGS.num_classes, is_training=False)

if FLAGS.eval_image_size is None:
  eval_image_size = network_fn.default_image_size
if type(eval_image_size) == int:
  eval_image_size = [eval_image_size, eval_image_size]


processed_image = image_preprocessing_fn(image,
                                         eval_image_size[0], eval_image_size[1])

processed_images  = tf.expand_dims(processed_image, 0) # Or tf.reshape(processed_image, (1, eval_image_size, eval_image_size, 3))

logits, _ = network_fn(processed_images)

probabilities = tf.nn.softmax(logits)

init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, slim.get_model_variables(model_variables))

sess = tf.Session()
init_fn(sess)

fout = sys.stdout
if FLAGS.outfile is not None:
  fout = open(FLAGS.outfile, 'a')
h = ['image']
h.extend(['class%s' % i for i in range(FLAGS.num_classes)])
h.append('predicted_class')
h.append('ground_truth_class')
h.append('correct')
print('\t'.join(h))


for fl in fls:
  image_name = None

  try:
    if FLAGS.tfrecord is False:
      x = tf.gfile.FastGFile(fl).read() # You can also use x = open(fl).read()
      image_name = os.path.basename(fl)
    else:
      example = tf.train.Example()
      example.ParseFromString(fl)

      # Note: The key of example.features.feature depends on how you generate tfrecord.
      x = example.features.feature['image/encoded'].bytes_list.value[0] # retrieve image string
      
      y = int(example.features.feature['image/class/label'].int64_list.value[0])
      image_name = 'TFRecord'

    probs = sess.run(probabilities, feed_dict={image_string:x})
    #np_image, network_input, probs = sess.run([image, processed_image, probabilities], feed_dict={image_string:x})

  except Exception as e:
    tf.logging.warn('Cannot process image file %s' % fl)
    continue

  probs = probs[0, 0:]
  a = [image_name]
  a.extend(probs)
  predicted_class = np.argmax(probs)
  a.append(predicted_class)
  a.append(y)
  a.append(y==predicted_class)
  print('\t'.join([str(e) for e in a]), file=fout)

sess.close()
fout.close()
