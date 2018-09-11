#!/usr/bin/env python3

'''class objects for dataset and samples'''

import numpy as np

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

def format_candidates(sample,corpus_data,can_list,men_list,men_padded,dict):
    logger.info('Formatting mentions...')
    sample.mentions, x_zero = _format_mentions_and_x0(can_list,men_list,men_padded)
    sample.x = _format_x(can_list,x_zero)

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
    return mentions, x_zero

def _format_x(can_list,x_zero,dict):
    '''
    can_list: list of list (mention level) of generated candidates
    x_zero: list of numpy array of vectorized mentions
    dict: vectorized controlled vocabulary
    '''
    # x[1] = vectorized candidates
    x_one = []
    debug_count_noncan = 0
    # x[2] = candidate generation scores
    x_two = []
    #can_list>mention format: list of lists of n (key,score) tuples
    for mention in can_list:
        for can, score in mention:
            x_one.append(dict.get(can,_non_canonical(can)))
            x_two.append([score])
        
            #check dict format, check terminology mapping
            #check how candidates are generated, if there's chance of non-canonical ids
    logger.debug('{0} non-canonical forms used.'.format(debug_count_noncan))

            #pad the thing



    x = np.array([x_zero,x_one,x_two])
    return x

def _non_canonical(id):
    assert id not in dict
    
        debug_count_noncan += 1
        return

    disambiguated = 


