#!/usr/bin/env python3

'''class objects for dataset and samples'''

import numpy as np
from keras.preprocessing.sequence import pad_sequences

import logging
logger = logging.getLogger(__name__)

class DataSet:
    def __init__(self):
        self.objects = None
        self.mentions = None
        self.tokenized_mentions = None
        self.vectorized_numpy_mentions = None
        self.padded = None

class Sample:
    '''
    format:
        generated: list of lists, each list contains
                   tuples of generated candidates, each
                   tuple contains ID and score
    '''
    def __init__(self):
        self.generated = None
        self.x = None
        self.y = None
        self.mentions = None

def format_candidates(sample,cor_mens,vec_dict):
    '''
    Formats candidates and assigns formatted candidates to sample's attributes
    Affected attributes: sample.mentions, sample.x

    Input:
        sample: object whose attribute is to be assigned
            sample.generated: list of generated candidates
        cor_mens: list of mentions
        vec_dict: vectorized controlled vocabulary
    '''
    logger.info('Formatting mentions...')
    logger.warning('Modify next line afterwards')
    sample.mentions, x_zero_np = _format_mentions_and_x0(sample.generated,cor_mens.mentions[:100],cor_mens.padded[:100])
    logger.debug('Seems fine up to here.')
    #sample.mentions, x_zero = _format_mentions_and_x0(sample.generated,cor_mens.mentions[:100],cor_mens.padded[:100])
    sample.x = _format_x(sample.generated,x_zero_np,vec_dict)
    assert len(sample.x[0])==len(sample.x[1]) & len(sample.x[1])==len(sample.x[2])

def _format_mentions_and_x0(can_list,men_list,men_padded):
    '''
    Input:
        can_list: list of generated candidates
        men_list: list of mentions
    Output:
        list of (start, end, mention), where end is not inclusive
        x_zero, vectorized mentions
    '''
    mentions = []
    start_index = 0
    end_index = 0
    x_zero = []
    assert len(can_list)==len(men_list) & len(men_list)==len(men_padded)
    for candidates, mention, padded_mention in zip(can_list,men_list,men_padded):
        can_number = len(candidates)
        x_zero.extend([padded_mention]*can_number)
        end_index = start_index + can_number
        mentions.append((start_index, end_index, mention))
        start_index = end_index
    return mentions, np.array(x_zero)
    #return mentions, x_zero

def _format_x(can_list,x_zero_np,vec_dict):
    '''
    can_list: list of list (mention level) of generated candidates in the format of
              (key,tokenized candidate,vectorized candidate,score)
    x_zero_np: np array of np array of vectorized mentions
    vec_dict: vectorized controlled vocabulary

    output: np.array(x_zero_padded,x_one_padded,x_two_padded)
    x_zero: np array of np array of vectorized mentions
    x_one: np array (mention in np array (vectorized candidates in np array))
    x_two: np array of np array of scores

    #check dict format: vectorized_dictionary[id]: np.array([[1, 1445], [1445, 1]]), vectorized AllNames
    #check terminology mapping
    #check how candidates are generated, if there's chance of non-canonical ids (no)
    '''
    # x[1] = vectorized candidates
    x_one = []
    # x[2] = candidate generation scores
    x_two = []
    logger.debug('Tokenized candidates not used for now, may be needed in the future.')
    #can_list>mention format: list of lists of n*(key,tokenized candidate,vectorized candidate,score)
    for mention in can_list:
        for candidate in mention:
            x_one.append(np.array(candidate[2]))
            x_two.append(np.array([candidate[3]]))
    '''
    Format of data at this point:
    x_zero: len = 2000, nparray of nparrays, padded
    x_one: len = 2000, list of nparrays, un-padded
    x_two: len = 2000, list of nparrays, no need to be padded
    '''
    logger.info('Padding...')
    x_one_padded = pad_sequences(np.array(x_one),padding='post',maxlen=len(max(x_one,key=len))) 
    x_two_np = np.array(x_two)
    logger.info('Padded shape: x[0]: {0}, x[1]: {1}, x[2]: {2}.'.format(x_zero_np.shape,x_one_padded.shape,x_two_np.shape))
    x = [x_zero_np,x_one_padded,x_two_np]
    return x
