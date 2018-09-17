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
        men_list: list of mentions
        vec_dict: vectorized controlled vocabulary
    '''
    logger.info('Formatting mentions...')
    logger.warning('Modify next line afterwards')
    sample.mentions, x_zero_np = _format_mentions_and_x0(sample.generated,cor_mens.mentions[:100],cor_mens.padded[:100])
    sample.x = _format_x(sample.generated,x_zero_np,vec_dict)

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
    for candidates, mention, padded_mention in zip(can_list,men_list,men_padded):
        can_number = len(candidates)
        x_zero.append(men_padded*can_number)
        end_index = start_index + can_number
        mentions.append(start_index, end_index, mention)
        start_index = end_index
    import pdb; pdb.set_trace()
    return mentions, np.array(x_zero)

def _format_x(can_list,x_zero_np,vec_dict):
    '''
    can_list: list of list (mention level) of generated candidates
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
    debug_count_noncan = 0
    # x[2] = candidate generation scores
    x_two = []
    #can_list>mention format: list of lists of n (key,score) tuples
    for mention in can_list:
        for can_id, score in mention:
            #x_one.append(vec_dict.get(can_id,_non_canonical(can_id)))
            x_one.append(vec_dict[can_id])
            x_two.append(np.array([score]))
    #logger.debug('{0} non-canonical forms used.'.format(debug_count_noncan))
    logger.info('Padding...')
    x_one_np = np.array(x_one)
    x_two_np = np.array(x_two)
    logger.info('Old shape: x[0]: {0}, x[1]: {1}, x[2]: {2}.'.format(x_zero_np.shape,x_one_np.shape,x_two_np.shape))
    x_zero_padded = pad_sequences(x_zero_np,padding='post', maxlen=len(max(x_zero_np,key=len)))
    x_one_padded = pad_sequences(x_one_np,padding='post')
    x_two_padded = pad_sequences(x_two_np,padding='post')
    logger.info('New shape: x[0]: {0}, x[1]: {1}, x[2]: {2}.'.format(x_zero_padded.shape,x_one_padded.shape,x_two_np.shape))
    x = np.array([x_zero,x_one,x_two])
    return x

'''
#not needed for now because all id are supposed to be canonical
#due to candidate_generation.generate_candidate function
def _non_canonical(can_id):
    assert can_id not in vec_dict
    
        debug_count_noncan += 1
        return
'''

