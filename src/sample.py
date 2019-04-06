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
        self.mention_ids = None
        self.elmo = None

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

class NewDataSet:
    '''
    class for run_new.py
    '''
    def __init__(self,info):
        self.info = info
        assert self.info in ['training corpus','dev corpus','test corpus','concepts']
        self.objects = None # freshly loaded in an ugly format

        self.ids = None
        self.all_ids = None # for dict concepts
        self.names = None
        self.map = None
        self.tokenize = None
        self.padded = None
        self.elmo = None

        self.vectorize = None # numpy array
        self.all = None # dict

    def info(self):
        return self.info

class Data:
    '''
    class for run_new.py
    '''
    def __init__(self):
        self.x = None
        self.y = None
        self.mentions = None

def canonical_id_list(id_list,dictionary_loaded,no_id_list):
    keys = dictionary_loaded.keys()
    cache = [_canonical(_nor_id(ID),keys,dictionary_loaded,no_id_list) for ID in id_list]
    return cache

def _nor_id(ID):
    '''
    The NCBI corpus provides ids in a format where
    MESH ids are in the form of 'DXXXXXX', whereas
    OMIM ids are in the form of 'OMIM:XXXXXX'.
    This function takes in a provided id STRING and
    returns the id STRING in the format used as keys
    in the control vocab.
    '''
    assert type(ID)==str
    # assert 'MESH' not in ID ## already tested
    if 'OMIM' not in ID:
        ID='MESH:'+ID.strip() #one single id had a space in front of it
    return ID

def _canonical(ID,keys,dictionary_loaded,no_id_list):
    '''
    This function takes in a provided id STRING and
    returns the id STRING in the canonical form.
    '''
    assert type(ID)==str
    cache = ID
    cache_list = []
    if ID not in keys:
        for k, v in dictionary_loaded.items():
            if ID in v.AllDiseaseIDs:
                cache_list.append(k)
        if len(cache_list) == 0:
            no_id_list.append(str(ID))
            return cache
        elif len(cache_list) > 1:
            logger.warning('{0} candidates for non-canonical ID {1}!'.format(len(cache_list),ID))
        ''' # tested, actually works
        else:
            logger.info('{0} changed to canonical id {1}'.format(ID,cache_list[0]))
        '''
        cache = cache_list[0]
    return cache

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
    sample.mentions, x_zero_np = _format_mentions_and_x0(sample.generated,cor_mens.mentions,cor_mens.padded)
    #sample.mentions, x_zero_np = _format_mentions_and_x0(sample.generated,cor_mens.mentions[:100],cor_mens.padded[:100])
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

def check_candidates(sample,ground_truths):
    '''
    sample: generated candidates, sample object whose 'generated' attribute is a
            list of tuples (id,tokenized_candidate,vectorized_candidate,score)
    ground_truths: list of lists of ids

    assigns list of numpy arrays of 0/1 to sample.y
    '''
    y = []
    for cans, gt in zip(sample.generated,ground_truths):
        for can in cans:
            '''
            can in format (id,tokenized_candidate,vectorized_candidate,score)
            gt in format ['MESH:D003110', 'MESH:D009369'] (multiple)
            '''
            #print('Can:',can[0],';\tGt:',gt)
            if can[0] == gt[0]:
                y.append(np.array([1]))
            else:
                y.append(np.array([0]))
    sample.y = np.array(y)

def sped_up_format_x(mentions,candidates):
    '''
    Input:
        mentions: pooled mentions
        candidates: pooled candidates
    [corpus.padded,len(corpus.padded)*[can_list.vectorized]
    '''
    can_no = len(candidates)
    men_no = len(mentions)
    x0 = [mention for mention in mentions for _ in range(can_no)]
    x1 = [candidate for _ in range(men_no) for candidate in candidates]
    assert len(x0)==len(x1)
    return [np.array(x0),np.array(x1)]

def no_cangen_format_x(mentions,candidates):
    '''
    Input:
        mentions: list of list of padded mentions
        candidates: list of list of padded candidates
    [corpus.padded,len(corpus.padded)*[can_list.vectorized]
    '''
    can_no = len(candidates)
    men_no = len(mentions)
    x0 = [mention for mention in mentions for _ in range(can_no)]
    x1 = [candidate for _ in range(men_no) for candidate in candidates]
    assert len(x0)==len(x1)
    logger.info('{0} of candidate-mention pairs generated'.format(len(x1)))
    from keras.preprocessing.sequence import pad_sequences
    x0 = pad_sequences(x0,padding='post')
    x1 = pad_sequences(x1,padding='post')
    return [x0,x1]

def no_cangen_format_mentions(mentions,can_no):
    '''
    Input:
        mentions: list of mentions
        can_no: no. of candidates
    Output: list of (start,end,'mention')
    '''
    mention_list = []
    start = 0
    end = can_no
    for mention in mentions:
        mention_list.append((start,end,mention))
        start = can_no + start
        end = can_no + end
    return mention_list

def no_cangen_format_y(candidates,ground_truths):
    '''
    Input:
        candidates:
        ground_truths: list of lists of ids

    returns list of numpy arrays of 0/1
    '''
    # golds = [item for sublist in ground_truths for item in sublist]
    import pdb;pdb.set_trace()
    print('condition of 1 and 0 being debugged')
    golds = [sublist[0] for sublist in ground_truths]
    # original line:
    # y = [[1] if gold == can else [0] for gold in golds for can in candidates]
    # haven't checked if this new line is correct, refer to gitig_test_evaluation.py
    y = [[1] if gold[0] in can and len(gold)==1 else [0] for gold in golds for can in candidates]
    y_ = [item for sublist in y for item in sublist]
    logger.debug('Total number of correct candidates: {0}'.format(sum(y_)))
    return np.array(y)

def load_no_cangen_data(pickled_objects,training_data,val_data):
    '''
    pickled_objects in the form of [[training_data.x,training_data.y,training_data.mentions],[val_data.x,val_data.y,val_data.mentions]]
    '''
    training_data.x = pickled_objects[0][0]
    training_data.y = pickled_objects[0][1]
    training_data.mentions = pickled_objects[0][2]
    val_data.x = pickled_objects[1][0]
    val_data.y = pickled_objects[1][1]
    val_data.mentions = pickled_objects[1][2]
    '''# did not work, could not assign
    for pickled_object,data in zip(pickled_objects,[training_data,val_data]):
        for o,d in zip(pickled_object,[data.x,data.y,data.mentions]):
            d = o[:]
    '''

def sample_format_mentions(sampled,corpus_names):
    '''
    Input:
        sampled: list of (picked_pos, list_of_sampled_neg)
    Output: list of (start,end,'mention')
    '''
    assert len(corpus_names) == len(sampled)
    mention_list = []
    start = 0
    end = 0
    for (picked_pos, sampled_neg),mention in zip(sampled,corpus_names):
        can_no = len(picked_pos) + len(sampled_neg)
        end = can_no + end
        mention_list.append((start,end,mention))
        start = end
    return mention_list
    
def _sample_format_x0(corpus_padded,formatted_mention):
    '''
    Input:
        formatted_mention: list of (start,end,'mention')
        corpus_padded: list of padded mentions
    '''
    assert len(formatted_mention) == len(corpus_padded)
    x0 = []
    for (start, end, tok),padded in zip(formatted_mention,corpus_padded):
        for i in range(end-start):
            x0.append(padded)
    return np.array(x0)

def _sample_format_x1(sampled,concept_padded):
    '''
    Input:
        sampled: list of (picked_pos, list_of_sampled_neg)
    '''
    #x1 = np.array([])
    #for picked_pos,sampled_neg in sampled:
    #    x1 = np.concatenate((x1, np.array(concept_padded)[np.array(picked_pos+sampled_neg)]),axis=0)
    x1 = []
    for picked_pos,sampled_neg in sampled:
        if picked_pos and sampled_neg:
            x1.extend(np.array(concept_padded)[np.array(picked_pos+sampled_neg)].tolist())
    return np.array(x1) 

def sample_format_x(sampled,corpus_padded,concept_padded,formatted_mention):
    x0 = _sample_format_x0(corpus_padded,formatted_mention)
    x1 = _sample_format_x1(sampled,concept_padded)
    assert len(x0)==len(x1)
    return [x0,x1]

def sample_format_y(sampled):
    y = []
    for picked_pos,sampled_neg in sampled:
        y.append([1]*len(picked_pos)+[0]*len(sampled_neg))
    y = [item for sublist in y for item in sublist]
    return np.array(y)

def examples(conf, concept, positives, vocab, neg_count=None):
    """
    Builds positive and negative examples.
    """
    if not neg_count:
        neg_count = conf.getint('sample','neg_count')
    while True:
        for (chosen_idx, idces), e_token_indices in positives:          
            if len(chosen_idx) ==1:
                # FIXME: only taking into account those that have exactly one gold concept
                c_token_indices = concept.vectorize[chosen_idx[0]]
            
                negative_token_indices = [concept.vectorize[i] for i in random.sample(list(set([*range(len(concept.names))])-set(idces)),neg_count)]

            entity_inputs = np.tile(pad_sequences([e_token_indices], padding='post', maxlen=conf.getint('embedding','length')), (len(negative_token_indices)+1, 1)) # Repeat the same entity for all concepts
            concept_inputs = pad_sequences([c_token_indices]+negative_token_indices, padding='post', maxlen=conf.getint('embedding','length'))
            # concept_inputs = np.asarray([[concept_dict[cid]] for cid in [concept_id]+negative_concepts])
            # import pdb; pdb.set_trace()
            distances = [1] + [0]*len(negative_token_indices)
            
            data = {
                'inp_mentions': entity_inputs,
                'inp_candidates': concept_inputs,
                'prediction_layer': np.asarray(distances),
            }
            
            yield data, data

def prepare_positives(positives,tokenizer,vocab):

    formatted = []
    for (chosen_idx,idces), span in positives:
        vec = [vocab.get(text.lower(),1) for text in tokenizer(span)]
        formatted.append(((chosen_idx,idces),vec))
    return formatted