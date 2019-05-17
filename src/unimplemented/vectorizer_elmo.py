#!/usr/bin/env python3
# coding: utf8

import numpy
import vectorizer

import logging
logger = logging.getLogger(__name__)
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

def elmo_default(context_lst_chunks):
	'''input format: list of list of this kind: ["the cat is on the mat", "dogs are in the fog"]'''
	with tf.Graph().as_default():
		logging.info('Adding module\'s variables to the current TensorFlow graph...')
		elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False) # trainable=True to train the 4 scalar weights (as described in the paper)

		# The default signature, the module takes untokenized sentences as input.
		# The input tensor is a string tensor with shape [batch_size].
		# The module tokenizes each string by splitting on spaces.
        
		count = 0
		logging.info('running %s tensorflow session(s)...', len(context_lst_chunks))
		for context_lst in context_lst_chunks:
			# Format of context_lst: ["the cat is on the mat", "dogs are in the fog"]
			embeddings = elmo(
			context_lst,
			signature="default",
			as_dict=False)    
			with tf.Session() as sess:
				sess.run(tf.global_variables_initializer())
				sess.run(tf.tables_initializer())
				#print(sess.run(embeddings))
				logging.info('%s done',str(count/len(context_lst_chunks)))
				count+= 1
				yield sess.run(embeddings)


'''
1. get the tokenized mentions into the wanted format
2. get the elmo representation of that
'''

def elmo_format_x(mentions,candidates):
    '''
    Input:
        mentions: list of mentions in elmo
        candidates: list of candidates in elmo
    '''
    can_no = len(candidates)
    men_no = len(mentions)
    x0 = [mention for mention in mentions for _ in range(can_no)]
    x1 = [candidate for _ in range(men_no) for candidate in candidates]
    assert len(x0)==len(x1)
    return [x0,x1]