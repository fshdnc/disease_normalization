#!/usr/bin/env python3
# coding: utf8

"""
A ranking accuracy callback.
Modified from: https://github.com/lfurrer/disease-normalization/blob/master/tzlink/rank/callback.py
"""
import numpy as np
import logging
logger = logging.getLogger(__name__)
print('change! callback.py, def evaluate after fixing logging')
from keras.callbacks import Callback

def evaluate(data_mentions, predictions, data_y):
	'''
	Input:
	data_mentions: e.g. val_data.mentions, of the form [(start,end,untok_mention),(),...,()]
	predictions: [[prob],[prob],...,[prob]]
	data_y: e.g. val_data.y, of the form [[0],[1],...,[0]]
	'''
	assert len(predictions) == len(data_y)
	correct = 0
	for start, end, untok_mention in data_mentions:
		index_prediction = np.argmax(predictions[start:end],axis=0)
		index_gold = np.argmax(data_y[start:end],axis=0)
		if index_prediction == index_gold:
			correct += 1
	total = len(data_mentions)
	accuracy = correct/total
	logger.info('Accuracy: {0}, Correct: {1}, Total: {2}'.format(accuracy,correct,total))
	print('Accuracy: {0}, Correct: {1}, Total: {2}'.format(accuracy,correct,total))
	return accuracy

class EarlyStoppingRankingAccuracy(Callback):
	''' Ranking accuracy callback with early stopping.

	'''
	def __init__(self, conf, val_data):
		super().__init__()

		self.conf = conf
		self.val_data = val_data

		self.best = 0 # best accuracy
		self.wait = 0
		self.stopped_epoch = 0
		self.patience = int(conf['training']['patience'])
		self.model_path = conf['model']['path_model_whole']

		self.save = int(self.conf['settings']['save_prediction'])

	def on_train_begin(self, logs={}):
		self.losses = []
		self.accuracy = []

		self.wait = 0
		return

	def on_epoch_end(self, epoch, logs={}):
		self.losses.append(logs.get('loss'))
		test_y = self.model.predict(self.val_data.x)
		evaluation_parameter = evaluate(self.val_data.mentions, test_y, self.val_data.y)
		self.accuracy.append(evaluation_parameter)

		if evaluation_parameter > self.best:
			self.best = evaluation_parameter
			self.model.save(self.model_path)
			self.wait = 0
			# something here to print trec_eval doc
		else:
			self.wait += 1
			if self.wait > int(self.conf['training']['patience']):
				self.stopped_epoch = epoch
				self.model.stop_training = True
		if self.save and self.model.stop_training:
			from model_tools import save_predictions
			logger.info('Saving predictions to {0}'.format(self.conf['model']['path_saved_predictions']))
			save_predictions(self.conf['model']['path_saved_predictions'],test_y) #(filename,predictions)
		print('Testing: epoch: {0}, self.model.stop_training: {1}'.format(epoch,self.model.stop_training))
		return

	def on_train_end(self, logs=None):
		if self.stopped_epoch > 0:
			logging.info('Epoch %05d: early stopping', self.stopped_epoch + 1)
		return

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))
		return