#!/usr/bin/env python3
# coding: utf8

import numpy

import logging
logger = logging.getLogger(__name__)

'''code copied from: https://machinelearningmastery.com/save-load-keras-deep-learning-models/'''

def save_model(model_object,model_file_name,weights_file_name):
	'''
	1. serialize model to JSON
	2. serialize weights to HDF5
	'''
	model_json = model_object.to_json()
	with open(model_file_name, "w") as json_file:
	    json_file.write(model_json)
	model_object.save_weights(weights_file_name)
	logger.info("Model saved.")

def load_model(model_file_name,weights_file_name, custom_layer_dict):
	'''
	1. load json and create model
	2. load weights into new model
	'''
	from keras.models import model_from_json
	json_file = open(model_file_name, 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json, custom_objects=custom_layer_dict)
	loaded_model.load_weights(weights_file_name)
	logger.info('Model loaded.')
	return loaded_model

def save_predictions(filename,predictions):
	import pickle
	with open(filename,'wb') as f:
		pickle.dump(predictions,f) #,protocol=4)

