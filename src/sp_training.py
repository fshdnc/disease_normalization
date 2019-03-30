#!/usr/bin/env python3
# coding: utf8

'''Sample for training'''

import numpy as np
import logging
logger = logging.getLogger(__name__)

from keras.callbacks import Callback
import random

def sample_hard_total(n, data_mentions, predictions, data_y):
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

	len_sample_not_n = 0
	oracle_zero = 0
	hard_n_negative = 0
	for start, end, untok_mention in data_mentions:
		oracle_index = [i for i,l in enumerate(data_y[start:end]) if l==np.array([1])]
		#print('oracle index:',oracle_index)
		non_oracle_index = [i for i,l in enumerate(data_y[start:end]) if l==np.array([0])]
		#print('non oracle index length',len(non_oracle_index))
		# assert len(oracle_index+non_oracle_index) == end-start
		hard_n = n - len(oracle_index)
		#print('hard_n',hard_n)
		non_oracle_predictions = np.array(predictions[start:end])[np.array(non_oracle_index)]
		#print('non_oracle_predictions length',len(non_oracle_predictions))
		hardest_index = np.argpartition(non_oracle_predictions,-hard_n)[-hard_n:].tolist()
		#print('hardest_index',hardest_index)
		#import pdb;pdb.set_trace()
		sampled = list(set(hardest_index+oracle_index))
		try:
			assert hard_n>0
		except AssertionError:
			hard_n_negative += 1
		try:
			assert len(sampled)==n
		except AssertionError:
			len_sample_not_n += 1
			continue # skips those with too many positive lavels
			#print(start,end,len(sampled),'\n',oracle_index,len(non_oracle_index),len(hardest_index))
		try:
			assert len(oracle_index)!=0
		except AssertionError:
			oracle_zero += 1
		hard_and_gold.extend([i+start for i in sampled])
	print('hard n is negative:',hard_n_negative)
	print('sample length does not equal to n:',len_sample_not_n)
	print('no oracle found:',oracle_zero)
	hard_and_gold.sort()
	assert len(hard_and_gold) == n*(len(data_mentions)-len_sample_not_n)
	return np.array(hard_and_gold)

# evaluate(self.val_data.mentions, test_y, self.val_data.y)

def sample_hard_ratio(r, z, data_mentions, predictions, data_y):
	'''
	Takes in predictions, gold standards, picks the oracles and hardest concepts,
	with the ratio of [1] and [0] being 1:r. If no [1], sample z.
	Returns (1) a list indexes of the sampled concepts
		    (2) a list of mentions [(start,end,untok_mention),(),...,()]

	Input:
	data_mentions: e.g. val_data.mentions, of the form [(start,end,untok_mention),(),...,()]
	predictions: [[prob],[prob],...,[prob]]
	data_y: e.g. val_data.y, of the form [[0],[1],...,[0]]
	'''
	assert len(predictions) == len(data_y)
	predictions = [item for sublist in predictions for item in sublist]
	hard_and_gold = []
	new_mentions = []

	oracle_zero = 0

	for start, end, untok_mention in data_mentions:
		oracle_index = []
		non_oracle_index = []
		for i,l in enumerate(data_y[start:end]):
			if l==np.array([1]):
				oracle_index.append(i)
			else:
				non_oracle_index.append(i)

		# assert len(oracle_index+non_oracle_index) == end-start
		if oracle_index:
			hard_n = len(oracle_index)*r
		else:
			hard_n = z
		#print('hard_n',hard_n)
		non_oracle_predictions = np.array(predictions[start:end])[np.array(non_oracle_index)]
		#print('non_oracle_predictions length',len(non_oracle_predictions))
		hardest_index = np.argpartition(non_oracle_predictions,-hard_n)[-hard_n:].tolist()
		#print('hardest_index',hardest_index)
		#import pdb;pdb.set_trace()
		sampled = list(set(hardest_index+oracle_index))
		try:
			assert len(oracle_index)!=0
		except AssertionError:
			oracle_zero += 1
		new_mention = (len(hard_and_gold),len(hard_and_gold)+len(sampled),untok_mention)

		hard_and_gold.extend([i+start for i in sampled])
		new_mentions.append(new_mention)
	print('no oracle found:',oracle_zero)
	hard_and_gold.sort()
	return np.array(hard_and_gold),new_mentions

def pick_positive_name(conf,corpus,concepts,idx):
	'''
	input: corpus obj, concept obj, index of corpus object
	return: index of picked concept name, the indices of other unpicked positives
	'''
	mention = corpus.names[idx]
	ID = corpus.ids[idx]

	correct_indices = []
	for i,v in enumerate(concepts.ids):
		# FIXME: not taking into account mentions with multiple mappings
		if v in ID:
			correct_indices.append(i)
	if correct_indices: # FIXME: 142 without canonical ids instead of 115
		correct = [*zip(correct_indices,np.array(concepts.names)[np.array(correct_indices)].tolist())]

		import difflib
		# FIXME: maybe use a better way to measure string similarity
		scores = [(i,difflib.SequenceMatcher(None, a=mention.lower(), b=name.lower()).ratio()) for i,name in correct]
		picked = [sorted(scores, key = lambda i: i[1],reverse=True)[0][0]]
		others = sorted([i for i, score in scores if i!=picked])

		# import random
		# sampled_neg = sorted(random.sample(list(set([*range(len(concepts.names))])-set(correct_indices)),conf.getint('sample','neg_count')))
		
	else: # ignore the ones whose IDs are not included in the vocab
		picked = []
		# sampled_neg = []
		others = []

	return (picked,others) #sampled_neg)

def sample_for_individual_mention(positives,no_of_concepts,neg_count):
	'''
	input: ([ind_of_picked],[indices of other unpicked positives])
	return: index of picked concept name, randomly sampled negatives
	'''
	if positives[0]:
		correct_indices = positives[0]+positives[1]
		sampled_neg = sorted(random.sample(list(set([*range(no_of_concepts)])-set(correct_indices)),neg_count))
	else:
		sampled_neg = []
	return (positives[0],sampled_neg)
