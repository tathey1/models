"""
author: Thomas Athey
date: 7/31/18

This class is for the sole purpose of creating a ditionary of variables
to be used as the var_list argument of tf.contrib.framework.assign_from_checkpoint_fn

"""

from __future__ import absolute_import, division, print_function
import tensorflow as tf
slim = tf.contrib.slim

def get_dict(variables_to_restore):
	varmap = {}
	vs = slim.get_model_variables()
	for path_var in vs:
		name = path_var.name
		for path_var_rest in variables_to_restore:
			if path_var_rest.name in name:
				new_name = name.replace('_final','')
				new_name = new_name[:-2] #remove the':0'
				print('Moving: ' + new_name)
                                print('to: ' + name)
				varmap[new_name] = path_var
	return varmap 
