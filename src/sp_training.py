#!/usr/bin/env python3
# coding: utf8

'''Sample for training'''

import numpy as np
import logging
logger = logging.getLogger(__name__)

from keras.callbacks import Callback

def sample_hard(n, data_mentions, predictions, data_y):
	'''
	Takes in predictions, gold standards, picks the oracles and hardest concepts,
	the numbers of which sum to n. Returns a list indexes of the sampled concepts.

	Input:
	data_mentions: e.g. val_data.mentions, of the form [(start,end,untok_mention),(),...,()]
	predictions: [[prob],[prob],...,[prob]]
	data_y: e.g. val_data.y, of the form [[0],[1],...,[0]]
	'''
	assert len(predictions) == len(data_y)
	predictions = [item for sublist in predictions for item in sublist]
	hard_and_gold = []
	for start, end, untok_mention in data_mentions:
		oracle_index = [i for i,l in enumerate(data_y[start:end]) if l==np.array([1])]
		non_oracle_index = [i for i,l in enumerate(data_y[start:end]) if l==np.array([0])]
		# assert len(oracle_index+non_oracle_index) == end-start
		hard_n = n - len(oracle_index)
		non_oracle_predictions = np.array(predictions[start:end])[np.array(non_oracle_index)]
		hardest_index = np.argpartition(non_oracle_predictions,-hard_n)[-hard_n:].tolist()
		#import pdb;pdb.set_trace()
		sampled = list(set(hardest_index+oracle_index))
		hard_and_gold.extend([i+start for i in sampled])
	hard_and_gold.sort()
	return np.array(hard_and_gold)

# evaluate(self.val_data.mentions, test_y, self.val_data.y)