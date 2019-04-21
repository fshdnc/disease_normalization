#!/usr/bin/env python3
# coding: utf8

"""
A ranking accuracy callback.
Modified from: https://github.com/lfurrer/disease-normalization/blob/master/tzlink/rank/callback.py
"""
import numpy as np
import logging
logger = logging.getLogger(__name__)

from keras.callbacks import Callback
from keras.models import load_model
from datetime import datetime
import io

import model_tools
from cnn import semantic_similarity_layer

def save_model(model, path,now):
	logger.info('Saving best model to {0}'.format(path+now))
	model_name = path + now + '.json'
	weights_name = path + now + '.h5'
	model_tools.save_model(model, model_name, weights_name)

def evaluate(data_mentions, predictions, data_y):
	'''
	Input:
	data_mentions: e.g. val_data.mentions, of the form [(start,end,untok_mention),(),...,()]
	predictions: [[prob],[prob],...,[prob]]
	data_y: e.g. val_data.y, of the form [[0],[1],...,[0]]
	'''
	assert len(predictions) == len(data_y)
	correct = 0
	logger.warning('High chance of same prediction scores.')
	for start, end, untok_mention in data_mentions:
		index_prediction = np.argmax(predictions[start:end],axis=0)
		# print(index_prediction) # prediction same for first few epochs
		if data_y[start:end][index_prediction] == 1:
			correct += 1
	total = len(data_mentions)
	accuracy = correct/total
	logger.info('Accuracy: {0}, Correct: {1}, Total: {2}'.format(accuracy,correct,total))
	return accuracy

def write_training_info(conf,path):
	import configparser
	with open(path,'w',encoding='utf-8') as configfile:    # save
		conf.write(configfile)

class Timed(Callback):
	''' 
	Calculates time taken.
	'''
	def __init__(self):
		super().__init__()
		self.before = None
		self.after = None
	def on_epoch_begin(self,epoch,logs={}):
		self.before = datetime.now()
	def on_epoch_end(self, epoch,logs={}):
		self.after = datetime.now()
		logger.info('Time taken for the epoch:{0}'.format(self.after-self.before))

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
		self.model_path = conf['model']['path_model_whole']

		self.save = int(self.conf['settings']['save_prediction'])
		self.now = datetime.now().strftime('%Y%m%d-%H%M%S')
		self.history = self.conf['settings']['history'] + self.now + '.txt'
		write_training_info(self.conf,self.history)

	def on_train_begin(self, logs={}):
		self.losses = []
		self.accuracy = []

		self.wait = 0
		with open(self.history,'a',encoding='utf-8') as fh:
		# Pass the file handle in as a lambda function to make it callable
			self.model.summary(print_fn=lambda x: fh.write(x + '\n'))
		return

	def on_epoch_end(self, epoch, logs={}):
		self.losses.append(logs.get('loss'))
		#before = datetime.now()
		test_y = self.model.predict(self.val_data.x)
		#after = datetime.now()
		#logger.info('Time taken for prediction without speedup:{0}'.format(after-before))
		evaluation_parameter = evaluate(self.val_data.mentions, test_y, self.val_data.y)
		self.accuracy.append(evaluation_parameter)
		with open(self.history,'a',encoding='utf-8') as f:
			f.write('Epoch: {0}, Training loss: {1}, validation accuracy: {2}\n'.format(epoch,logs.get('loss'),evaluation_parameter))

		if evaluation_parameter > self.best:
			logging.info('Intermediate model saved.')
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
			logger.info('Saving predictions to {0}'.format(self.conf['model']['path_saved_predictions']))
			model_tools.save_predictions(self.conf['model']['path_saved_predictions'],test_y) #(filename,predictions)
		logger.info('Testing: epoch: {0}, self.model.stop_training: {1}'.format(epoch+1,self.model.stop_training))
		return

	def on_train_end(self, logs=None):
		if self.stopped_epoch > 0:
			logging.info('Epoch %05d: early stopping', self.stopped_epoch + 1)
		if self.conf.getint('model','save'):
			self.model = load_model(self.model_path,custom_objects={'semantic_similarity_layer': semantic_similarity_layer})
			save_model(self.model, self.conf['model']['path'],self.now)
		return

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))
		return

class EarlyStoppingRankingAccuracySpedUp(Callback):
	''' Ranking accuracy callback with early stopping.

	'''
	def __init__(self, conf, val_data, concept_padded, corpus_padded,pretrained):
		super().__init__()

		self.conf = conf
		self.val_data = val_data
		self.concept_padded = concept_padded
		self.corpus_padded = corpus_padded
		self.pretrained = pretrained
		self.convoluted_input = None
		self.prediction_model = None

		self.best = 0 # best accuracy
		self.wait = 0
		self.stopped_epoch = 0
		self.model_path = conf['model']['path_model_whole']

		self.save = int(self.conf['settings']['save_prediction'])
		self.now = datetime.now().strftime('%Y%m%d-%H%M%S')
		self.history = self.conf['settings']['history'] + self.now + '.txt'
		write_training_info(self.conf,self.history)

	def on_train_begin(self, logs={}):
		self.losses = []
		self.accuracy = []

		self.wait = 0
		with open(self.history,'a',encoding='utf-8') as fh:
		# Pass the file handle in as a lambda function to make it callable
			self.model.summary(print_fn=lambda x: fh.write(x + '\n'))
		return

	def on_epoch_end(self, epoch, logs={}):
		self.losses.append(logs.get('loss'))

		from cnn import forward_pass_speedup
		before = datetime.now()
		self.convoluted_input, self.prediction_model = forward_pass_speedup(self.model,self.corpus_padded,self.concept_padded,self.pretrained)
		test_y = self.prediction_model.predict(self.convoluted_input)
		after = datetime.now()
		logger.info('Time taken for prediction with speedup:{0}'.format(after-before))
		evaluation_parameter = evaluate(self.val_data.mentions, test_y, self.val_data.y)
		self.accuracy.append(evaluation_parameter)
		self.convoluted_input = None
		self.prediction_model = None
		with open(self.history,'a',encoding='utf-8') as f:
			f.write('Epoch: {0}, Training loss: {1}, validation accuracy: {2}\n'.format(epoch,logs.get('loss'),evaluation_parameter))


		if evaluation_parameter > self.best:
			logging.info('Intermediate model saved.')
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
			logger.info('Saving predictions to {0}'.format(self.conf['model']['path_saved_predictions']))
			model_tools.save_predictions(self.conf['model']['path_saved_predictions'],test_y) #(filename,predictions)
		logger.info('Testing: epoch: {0}, self.model.stop_training: {1}'.format(epoch,self.model.stop_training))
		return

	def on_train_end(self, logs=None):
		if self.stopped_epoch > 0:
			logging.info('Epoch %05d: early stopping', self.stopped_epoch + 1)
		if self.conf.getint('model','save'):
			self.model = load_model(self.model_path,custom_objects={'semantic_similarity_layer': semantic_similarity_layer})
			save_model(self.model, self.conf['model']['path'],self.now)
		return

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))
		return


class EarlyStoppingRankingAccuracySpedUpSharedEncoder(Callback):
	''' Ranking accuracy callback with early stopping.

	'''
	def __init__(self, conf, val_data, concept_padded, corpus_padded, pretrained):
		super().__init__()

		self.conf = conf
		self.val_data = val_data
		self.concept_padded = concept_padded
		self.corpus_padded = corpus_padded
		self.pretrained = pretrained
		self.convoluted_input = None
		self.prediction_model = None

		self.best = 0 # best accuracy
		self.wait = 0
		self.stopped_epoch = 0
		self.model_path = conf['model']['path_model_whole']

		self.save = int(self.conf['settings']['save_prediction'])
		self.now = datetime.now().strftime('%Y%m%d-%H%M%S')
		self.history = self.conf['settings']['history'] + self.now + '.txt'
		write_training_info(self.conf,self.history)

	def on_train_begin(self, logs={}):
		self.losses = []
		self.accuracy = []

		self.wait = 0
		with open(self.history,'a',encoding='utf-8') as fh:
		# Pass the file handle in as a lambda function to make it callable
			self.model.summary(print_fn=lambda x: fh.write(x + '\n'))
		return

	def on_epoch_end(self, epoch, logs={}):
		self.losses.append(logs.get('loss'))

		from cnn import forward_pass_speedup_shared_encoder
		before = datetime.now()
		self.convoluted_input, self.prediction_model = forward_pass_speedup_shared_encoder(self.model,self.corpus_padded,self.concept_padded,self.pretrained)
		test_y = self.prediction_model.predict(self.convoluted_input)
		after = datetime.now()
		logger.info('Time taken for prediction with speedup:{0}'.format(after-before))
		evaluation_parameter = evaluate(self.val_data.mentions, test_y, self.val_data.y)
		self.accuracy.append(evaluation_parameter)
		self.convoluted_input = None
		self.prediction_model = None
		with open(self.history,'a',encoding='utf-8') as f:
			f.write('Epoch: {0}, Training loss: {1}, validation accuracy: {2}\n'.format(epoch,logs.get('loss'),evaluation_parameter))


		if evaluation_parameter > self.best:
			logging.info('Intermediate model saved.')
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
			logger.info('Saving predictions to {0}'.format(self.conf['model']['path_saved_predictions']))
			model_tools.save_predictions(self.conf['model']['path_saved_predictions'],test_y) #(filename,predictions)
		logger.info('Testing: epoch: {0}, self.model.stop_training: {1}'.format(epoch,self.model.stop_training))
		return

	def on_train_end(self, logs=None):
		if self.stopped_epoch > 0:
			logging.info('Epoch %05d: early stopping', self.stopped_epoch + 1)
		if self.conf.getint('model','save'):
			self.model.load_weights(self.model_path)
			save_model(self.model, self.conf['model']['path'],self.now)
		return

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))
		return

class EarlyStoppingRankingAccuracySpedUpGiveModel(Callback):
	''' Ranking accuracy callback with early stopping.

	'''
	def __init__(self, conf, val_data, concept_padded, corpus_padded, pretrained, create_spedup_model):
		super().__init__()

		self.conf = conf
		self.val_data = val_data
		self.concept_padded = concept_padded
		self.corpus_padded = corpus_padded
		self.pretrained = pretrained
		self.convoluted_input = None
		self.prediction_model = None
		self.create_spedup_model = create_spedup_model

		self.best = 0 # best accuracy
		self.wait = 0
		self.stopped_epoch = 0
		self.model_path = conf['model']['path_model_whole']

		self.save = int(self.conf['settings']['save_prediction'])
		self.now = datetime.now().strftime('%Y%m%d-%H%M%S')
		self.history = self.conf['settings']['history'] + self.now + '.txt'
		write_training_info(self.conf,self.history)

	def on_train_begin(self, logs={}):
		self.losses = []
		self.accuracy = []

		self.wait = 0
		with open(self.history,'a',encoding='utf-8') as fh:
		# Pass the file handle in as a lambda function to make it callable
			self.model.summary(print_fn=lambda x: fh.write(x + '\n'))
		return

	def on_epoch_end(self, epoch, logs={}):
		self.losses.append(logs.get('loss'))
		before = datetime.now()
		self.convoluted_input, self.prediction_model = self.create_spedup_model(self.model,self.corpus_padded,self.concept_padded,self.pretrained)
		test_y = self.prediction_model.predict(self.convoluted_input)
		after = datetime.now()
		logger.debug('Time taken for prediction with speedup:{0}'.format(after-before))
		evaluation_parameter = evaluate(self.val_data.mentions, test_y, self.val_data.y)
		self.accuracy.append(evaluation_parameter)
		self.convoluted_input = None
		self.prediction_model = None
		with open(self.history,'a',encoding='utf-8') as f:
			f.write('Epoch: {0}, Training loss: {1}, validation accuracy: {2}\n'.format(epoch,logs.get('loss'),evaluation_parameter))


		if evaluation_parameter > self.best:
			logging.info('Intermediate model saved.')
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
			logger.info('Saving predictions to {0}'.format(self.conf['model']['path_saved_predictions']))
			model_tools.save_predictions(self.conf['model']['path_saved_predictions'],test_y) #(filename,predictions)
		logger.info('Testing: epoch: {0}, self.model.stop_training: {1}'.format(epoch,self.model.stop_training))
		return

	def on_train_end(self, logs=None):
		if self.stopped_epoch > 0:
			logging.info('Epoch %05d: early stopping', self.stopped_epoch + 1)
		try:
			self.model.load_weights(self.model_path)
		except OSError:
			pass
		# function in run_generator
		# predict(self.conf, self.concept, self.positives, self.vocab, self.entity_model, self.concept_model,self.model, self.val_data, result=self.history)
		if self.conf.getint('model','save'):
			save_model(self.model, self.conf['model']['path'],self.now)
		return

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))
		return


class EarlyStoppingRankingAccuracyGenerator(Callback):
    ''' Ranking accuracy callback with early stopping.

    '''
    def __init__(self, conf, concept, positives, vocab, entity_model, concept_model, original_model,val_data):
        super().__init__()
        self.conf = conf
        self.concept = concept
        self.positives = positives
        self.vocab = vocab
        self.entity_model = entity_model
        self.concept_model = concept_model
        self.original_model = original_model
        self.val_data = val_data

        self.best = 0 # best accuracy
        self.wait = 0
        self.stopped_epoch = 0
        self.patience = int(conf['training']['patience'])
        self.model_path = conf['model']['path_model_whole']

        self.save = int(self.conf['settings']['save_prediction'])
        self.now = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.history = self.conf['settings']['history'] + self.now + '.txt'
        write_training_info(self.conf,self.history)

    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []

        self.wait = 0
        with open(self.history,'a',encoding='utf-8') as fh:
        # Pass the file handle in as a lambda function to make it callable
            self.original_model.summary(print_fn=lambda x: fh.write(x + '\n'))
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))

        evaluation_parameter = predict(self.conf, self.concept, self.positives, self.vocab, self.entity_model, self.concept_model,self.model, self.val_data)
        self.accuracy.append(evaluation_parameter)

        with open(self.history,'a',encoding='utf-8') as f:
            f.write('Epoch: {0}, Training loss: {1}, validation accuracy: {2}\n'.format(epoch,logs.get('loss'),evaluation_parameter))

        if evaluation_parameter > self.best:
            logging.info('Intermediate model saved.')
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
            logger.info('Saving predictions to {0}'.format(self.conf['model']['path_saved_predictions']))
            model_tools.save_predictions(self.conf['model']['path_saved_predictions'],test_y) #(filename,predictions)
        logger.info('Testing: epoch: {0}, self.model.stop_training: {1}'.format(epoch,self.model.stop_training))
        return

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            logging.info('Epoch %05d: early stopping', self.stopped_epoch + 1)
        try:
            self.model.load_weights(self.model_path)
        except OSError:
            pass
        # function in run_generator
        # predict(self.conf, self.concept, self.positives, self.vocab, self.entity_model, self.concept_model,self.model, self.val_data, result=self.history)
        if self.conf.getint('model','save'):
            save_model(self.model, self.conf['model']['path'],self.now)
        return

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        return