'''
Thesis: 7.6 Evaluation on held-out data
N.B. This code requires saved model to run

Evaluate given model on
1. sampled development set
2. development set
3. test set

first argument: time stamp of the model e.g. 20190512-215143
second argument: filter_number
third argument: kernel_size
'''


import sys

import random
random.seed(1)

import os
import time
import logging
import logging.config

import configparser as cp

import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pickle
import nltk

import vectorizer
import load
import sample


#configurations
TIME = time.strftime('%Y%m%d-%H%M%S')
dynamic_defaults = {'timestamp': TIME}
config = cp.ConfigParser(defaults=dynamic_defaults,interpolation=cp.ExtendedInterpolation(),strict=False)
try:
    directory = os.path.join(os.path.abspath(os.path.dirname(__file__)))
    config.read(os.path.join(directory, 'fe.cfg'))
except NameError:
    directory = '/home/lhchan/disease_normalization/src'
    config.read(os.path.join(directory, 'fe.cfg'))

# saved model, change model_name to your own path
model_name = '/home/lhchan/disease_normalization/gitig/model_whole_'+sys.argv[1]+'.h5'
#################################################
config['embedding']['emb_file'] = os.path.join(directory, '../../disease_normalization/data/pubmed2018_w2v_200D/pubmed2018_w2v_200D.bin')
config['terminology']['dict_file'] = os.path.join(directory, '../../old-disease-normalization/data/ncbi-disease/CTD_diseases.tsv')
config['corpus']['training_file'] = os.path.join(directory,'../../old-disease-normalization/data/ncbi-disease/NCBItestset_corpus.txt')
config['corpus']['development_file'] = os.path.join(directory,'../../old-disease-normalization/data/ncbi-disease/NCBIdevelopset_corpus.txt')

config['cnn']['filters'] = sysargv2
config['cnn']['kernel_size'] = sysargv3
config['note']['note'] = 'final experiment, final numbers, model:' + TIME
#################################################
if config.getint('settings','gpu'):
    import tensorflow as tf
    gpu_config = tf.ConfigProto()
    gpu_config.gpu_options.allow_growth=True
    sess = tf.Session(config=gpu_config)


#logging
logger = logging.getLogger(__name__)
logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level':'INFO',
            'formatter': 'standard',
            'class':'logging.StreamHandler',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': True
        }
    }
})


# word embedding
vector_model, vocabulary, inversed_vocabulary = vectorizer.prepare_embedding_vocab(config['embedding']['emb_file'], binary = True, limit = config.getint('embedding','limit'))
pretrained = vectorizer.load_pretrained_word_embeddings(vocabulary, vector_model)

# MEDIC dictionary
dictionary = load.Terminology()
# dictionary of entries, key = canonical id, value = named tuple in the form of
#   MEDIC_ENTRY(DiseaseID='MESH:D005671', DiseaseName='Fused Teeth',
#   AllDiseaseIDs=('MESH:D005671',), AllNames=('Fused Teeth', 'Teeth, Fused')
dictionary.loaded = load.load(os.path.normpath(config['terminology']['dict_file']),'MEDIC')


def concept_obj(conf,dictionary,order=None):
    concept_ids = [] # list of all concept ids
    #concept_all_ids = [] # list of (lists of all concept ids with alt IDs)
    concept_names = [] # list of all names, same length as concept_ids
    concept_map = {} # names as keys, ids as concepts

    if order:
        use = order
        logger.info('Re-initializing concept object.')
    else:
        use = dictionary.loaded.keys()

    for k in use:
    # keys not in congruent order! To make them congruent:
    # k,v = zip(*dictionary.loaded.items())
    # k = list(k)
    # k.sort()
        c_id = dictionary.loaded[k].DiseaseID
        #a_ids = dictionary.loaded[k].AllDiseaseIDs
        
        for n in dictionary.loaded[k].AllNames:
            concept_ids.append(c_id)
            #concept_all_ids.append(a_ids)
            concept_names.append(n)
            if n in concept_map: # one name corresponds to multiple concepts
                concept_map[n].append(c_id)
                # logger.warning('{0} already in the dictionary with id {1}'.format(n,concept_map[n]))
            else:
                concept_map[n] = [c_id]

    # save the stuff to object
    concept = sample.NewDataSet('concepts')
    concept.ids = concept_ids
    #concept.all_ids = concept_all_ids
    concept.names = concept_names
    concept.map = concept_map
    concept.tokenize = [nltk.word_tokenize(name) for name in concept_names]
    concept.vectorize = np.array([[vocabulary.get(text.lower(),1) for text in concept] for concept in concept.tokenize])
    concept.padded = pad_sequences(concept.vectorize, padding='post', maxlen=int(config['embedding']['length']))
    return concept


# # validation set
# [real_val_data,concept_order] = pickle.load(open(os.path.join(directory, 'gitig_real_val_data.pickle'),'rb'))
# real_val_data.y=np.array(real_val_data.y)
# real_val_data.x = None

logger.info('Using truncated development corpus for evaluation.')
#corpus_dev = sample.NewDataSet('dev corpus')
[real_val_data,concept_order,corpus_dev] = pickle.load(open(os.path.join(directory, 'gitig_real_val_data_truncated_d50_p5.pickle'),'rb'))
real_val_data.y=np.array(real_val_data.y)

concept = concept_obj(config,dictionary,order=concept_order)    


from sample import prepare_positives,examples
positives_training, positives_dev, positives_dev_truncated = pickle.load(open(os.path.join(directory, 'gitig_positive_indices.pickle'),'rb'))
# positives_dev = prepare_positives(positives_dev,nltk.word_tokenize,vocabulary)
positives_dev_truncated = prepare_positives(positives_dev_truncated,nltk.word_tokenize,vocabulary)
del positives_dev, positives_training


# corpus
# corpus_train = sample.NewDataSet('training corpus')
# corpus_train.objects = load.load(os.path.normpath(config['corpus']['training_file']),'NCBI')

corpus_dev = sample.NewDataSet('dev corpus')
corpus_dev.objects = load.load(config['corpus']['development_file'],'NCBI')

for corpus in [corpus_dev]:
    corpus.ids = [] # list of all ids (gold standard for each mention)
    corpus.names = [] # list of all names
    corpus.all = [] # list of tuples (mention_text,gold,context,(start,end,docid))

    #sth wrong here that sometimes throw an error
    #import pdb;pdb.set_trace()
    for abstract in corpus.objects:
        for section in abstract.sections: # title and abstract
            for mention in section.mentions:
                nor_ids = [sample._nor_id(one_id) for one_id in mention.id]
                corpus.ids.append(nor_ids) # append list of ids, usually len(list)=1
                corpus.names.append(mention.text)
                corpus.all.append((mention.text,nor_ids,section.text,(mention.start,mention.end,abstract.docid)))
    #corpus.tokenize = [nltk.word_tokenize(name) for name in corpus.names]
    #corpus.vectorize = np.array([[vocabulary.get(text.lower(),1) for text in mention] for mention in corpus.tokenize])
    #corpus.padded = pad_sequences(corpus.vectorize, padding='post', maxlen=int(config['embedding']['length']))
    #corpus.tokenize = None
    #corpus.vectorize = None


# sampling
def examples(config, concept, positives, vocab, neg_count=config.getint('sample','neg_count')):
    """
    Builds positive and negative examples.
    """
    while True:
        for (chosen_idx, idces), e_token_indices in positives:          
            if len(chosen_idx) ==1:
                # FIXME: only taking into account those that have exactly one gold concept
                c_token_indices = concept.vectorize[chosen_idx[0]]
            
                negative_token_indices = [concept.vectorize[i] for i in random.sample(list(set([*range(len(concept.names))])-set(idces)),neg_count)]

                entity_inputs = np.tile(pad_sequences([e_token_indices], padding='post', maxlen=config.getint('embedding','length')), (len(negative_token_indices)+1, 1)) # Repeat the same entity for all concepts
                concept_inputs = pad_sequences([c_token_indices]+negative_token_indices, padding='post', maxlen=config.getint('embedding','length'))
                # concept_inputs = np.asarray([[concept_dict[cid]] for cid in [concept_id]+negative_concepts])
                # import pdb; pdb.set_trace()
                distances = [1] + [0]*len(negative_token_indices)

                data = {
                    'inp_mentions': entity_inputs,
                    'inp_candidates': concept_inputs,
                    'prediction_layer': np.asarray(distances),
                }
            
                yield data, data

def examples_evaluation(config, concept, positives, vocab):
    """
    Builds positive and negative examples.
    """
    while True:
        for (chosen_idx, idces), e_token_indices in positives:     
            entity_inputs = pad_sequences([e_token_indices], padding='post', maxlen=config.getint('embedding','length'))

            if len(chosen_idx) ==1: # the distances doesn't really matter for the evaluation
                distances = [1]
            else:
                distances = [0]
            data = {
                'inp_mentions': entity_inputs,
                'prediction_layer': np.asarray(distances),
            }
            
            yield data, data


def evaluate_w_results(data_mentions, predictions, data_y, concept, history):
    '''
    Input:
    data_mentions: e.g. val_data.mentions, of the form [(start,end,untok_mention),(),...,()]
    predictions: [[prob],[prob],...,[prob]]
    data_y: e.g. val_data.y, of the form [[0],[1],...,[0]]
    '''
    assert len(predictions) == len(data_y)
    correct = 0
    f = open(history,"a",encoding='utf-8')
    for start, end, untok_mention in data_mentions:
        index_prediction = np.argmax(predictions[start:end],axis=0)
        if data_y[start:end][index_prediction] == 1:
        ##index_gold = np.argmax(data_y[start:end],axis=0)
        ##if index_prediction == index_gold:
            correct += 1
            f.write('Correct - Gold: {0}, Prediction: {1}\n'.format(untok_mention,concept.names[index_prediction.tolist()[0]]))
        else:
            f.write('Incorrect - Gold: {0}, Prediction: {1}\n'.format(untok_mention,concept.names[index_prediction.tolist()[0]]))
    total = len(data_mentions)
    accuracy = correct/total
    f.write('Accuracy: {0}, Correct: {1}, Total: {2}'.format(accuracy,correct,total))
    f.close()
    return accuracy


from keras.models import Model
from keras.layers import Input, Embedding, Concatenate, Dense, Conv1D, GlobalMaxPooling1D, Flatten
from keras import layers
from cnn import semantic_similarity_layer
import callback

def _predict_shared_encoder_dot(original_model,entity_encodings,concept_encodings):
    layerss = ['v_sem','hidden_layer','prediction_layer']
    d1 = original_model.get_layer(layerss[1])
    d2 = original_model.get_layer(layerss[2])
    sim = layers.dot([entity_encodings, concept_encodings], axes=-1, normalize=True)
    concatenate_list = [entity_encodings,concept_encodings,sim]
    join_layer = Concatenate()(concatenate_list)
    hidden_layer = Dense(d1.units, activation=d1.activation,weights=d1.get_weights())(join_layer)
    prediction_layer = Dense(d2.units, activation=d2.activation,weights=d2.get_weights())(hidden_layer)
    model = Model(inputs=[entity_encodings,concept_encodings], outputs=prediction_layer)
    return model

def predict(config, concept, positives, vocab, entity_model, concept_model, original_model,val_data,result=None):
    entity_examples = examples_evaluation(config, concept, positives, vocab)

    c_token_indices = [[vocab.get(t.lower(), 1) for t in nltk.word_tokenize(neg)] for neg in concept.names]
    concept_examples = pad_sequences(c_token_indices, padding='post', maxlen=config.getint('embedding','length'))
    
    entity_encodings = entity_model.predict_generator(entity_examples, steps=len(positives))    
    concept_encodings = concept_model.predict(concept_examples)

    ###################
    from sample import sped_up_format_x
    convoluted_input = sped_up_format_x(entity_encodings,concept_encodings)

    entity_encodings = Input(shape=(convoluted_input[0].shape[1],),dtype='float32', name='entity_encodings')
    concept_encodings = Input(shape=(convoluted_input[1].shape[1],),dtype='float32', name='concept_encodings')

    model = _predict_shared_encoder_dot(original_model,entity_encodings,concept_encodings)

    test_y = model.predict(convoluted_input)

    if not result:
        evaluation_parameter = callback.evaluate(val_data.mentions, test_y, val_data.y)
    else:
        evaluation_parameter = evaluate_w_results(val_data.mentions, test_y, val_data.y, concept, result)

    return evaluation_parameter


from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from keras.callbacks import Callback
from model_tools import load_model, save_model
import cnn, model_tools, callback

tr_data = 'dummy'
model_shared_encoder_dot , entity_model, concept_model = cnn.build_model_shared_encoder_dot(config,tr_data,vocabulary,pretrained)
model_shared_encoder_dot.load_weights(model_name)

file_name = '/home/lhchan/disease_normalization/gitig/log/' + TIME + '.txt'

with open(file_name,'a',encoding='utf-8') as fh:
# Pass the file handle in as a lambda function to make it callable
    model_shared_encoder_dot.summary(print_fn=lambda x: fh.write(x + '\n'))

evaluation_parameter = predict(config, concept, positives_dev_truncated, vocabulary, entity_model, concept_model, model_shared_encoder_dot, real_val_data, result=file_name)
print('Sampled dev set accuracy:')
print(evaluation_parameter)

positives_training, positives_dev, positives_dev_truncated = pickle.load(open(os.path.join(directory, 'gitig_positive_indices.pickle'),'rb'))
positives_training = prepare_positives(positives_training,nltk.word_tokenize,vocabulary)
positives_dev = prepare_positives(positives_dev,nltk.word_tokenize,vocabulary)
#positives_dev_truncated = prepare_positives(positives_dev_truncated,nltk.word_tokenize,vocabulary)
del positives_dev_truncated

# validation set
[real_val_data,concept_order] = pickle.load(open(os.path.join(directory, 'gitig_real_val_data.pickle'),'rb'))
real_val_data.y=np.array(real_val_data.y)
real_val_data.x = None

evaluation_parameter = predict(config, concept, positives_dev, vocabulary, entity_model, concept_model, model_shared_encoder_dot, real_val_data, result=file_name)
print('Dev set accuracy:')
print(evaluation_parameter)


corpus_test = sample.NewDataSet('dev corpus')
corpus_test.objects = load.load(os.path.normpath(config['corpus']['training_file']),'NCBI')

for corpus in [corpus_test]:
    corpus.ids = [] # list of all ids (gold standard for each mention)
    corpus.names = [] # list of all names
    corpus.all = [] # list of tuples (mention_text,gold,context,(start,end,docid))

    #sth wrong here that sometimes throw an error
    #import pdb;pdb.set_trace()
    for abstract in corpus.objects:
        for section in abstract.sections: # title and abstract
            for mention in section.mentions:
                nor_ids = [sample._nor_id(one_id) for one_id in mention.id]
                corpus.ids.append(nor_ids) # append list of ids, usually len(list)=1
                corpus.names.append(mention.text)
                corpus.all.append((mention.text,nor_ids,section.text,(mention.start,mention.end,abstract.docid)))
    #corpus.tokenize = [nltk.word_tokenize(name) for name in corpus.names]
    #corpus.vectorize = np.array([[vocabulary.get(text.lower(),1) for text in mention] for mention in corpus.tokenize])
    #corpus.padded = pad_sequences(corpus.vectorize, padding='post', maxlen=int(config['embedding']['length']))
    #corpus.tokenize = None
    #corpus.vectorize = None

positives_test = pickle.load(open(os.path.join(directory, 'gitig_positive_indices_test.pickle'),'rb'))
positives_test = prepare_positives(positives_test,nltk.word_tokenize,vocabulary)

[tst_data,concept_order] = pickle.load(open(os.path.join(directory, 'gitig_test_data.pickle'),'rb'))
tst_data.y=np.array(tst_data.y)

evaluation_parameter = predict(config, concept, positives_test, vocabulary, entity_model, concept_model, model_shared_encoder_dot, tst_data, result=file_name)
print('Test set accuracy:')
print(evaluation_parameter)