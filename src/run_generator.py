'''
Shared encoder, generator
'''


import logging
import logging.config

import configparser as cp
import args

import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pickle
import nltk

import vectorizer
import load
import sample

#configurations
config = cp.ConfigParser(strict=False)
config.read('defaults.cfg')
#################################################
config['embedding']['emb_file'] = '/home/lenz/disease-normalization/data/embeddings/wvec_50_haodi-li-et-al.bin'
config['cnn']['filters'] = '20'
config['cnn']['optimizer'] = 'adam'
config['cnn']['lr'] = '0.0001'
config['cnn']['loss'] = 'binary_crossentropy'
config['cnn']['dropout'] = '0.3'
config['embedding']['length'] = '5'
config['embedding']['limit'] = '1000000'
config['note']['note'] = 'change to generator, d=50, p=5'
#################################################

#argparser
args = args.get_args()
'''
>>> args.train
False
'''

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
dictionary.loaded = load.load(config['terminology']['dict_file'],'MEDIC')


def concept_obj(conf,dictionary,order=None):
    concept_ids = [] # list of all concept ids
    # concept_all_ids = [] # list of (lists of all concept ids with alt IDs)
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
        # a_ids = dictionary.loaded[k].AllDiseaseIDs
        
        if int(conf['settings']['all_names']):
            for n in dictionary.loaded[k].AllNames:
                concept_ids.append(c_id)
                # concept_all_ids.append(a_ids)
                concept_names.append(n)
                if n in concept_map: # one name corresponds to multiple concepts
                    concept_map[n].append(c_id)
                    # logger.warning('{0} already in the dictionary with id {1}'.format(n,concept_map[n]))
                else:
                    concept_map[n] = [c_id]
        else:
            for n in dictionary.loaded[k].DiseaseName:
                concept_ids.append(c_id)
                concept_all_ids.append(a_ids)
                concept_names.append(n)
                if n in concept_map: # one name corresponds to multiple concepts
                    concept_map[n].append(c_id)
                    # logger.warning('{0} already in the dictionary with id {1}'.format(n,concept_map[n]))
                else:
                    concept_map[n] = [c_id]

    # save the stuff to object
    concept = sample.NewDataSet('concepts')
    concept.ids = concept_ids
    # concept.all_ids = concept_all_ids
    concept.names = concept_names
    concept.map = concept_map

    return concept


[val_data_truncated,concept_order, corpus_dev_truncated] = pickle.load(open('gitig_real_val_data_truncated_d50_p5.pickle','rb'))
val_data_truncated.y=np.array(val_data_truncated.y)
#corpus_dev_truncated.padded = pad_sequences(corpus_dev_truncated.vectorize, padding='post', maxlen=int(config['embedding']['length']))
corpus_dev_truncated.tokenize = None
corpus_dev_truncated.vectorize = None

concept = concept_obj(config,dictionary,order=concept_order)


# corpus
corpus_train = sample.NewDataSet('training corpus')
corpus_train.objects = load.load(config['corpus']['training_file'],'NCBI')

for corpus in [corpus_train]:
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


# sample concepts for each mention
import sp_training, sample
logger.info('Sampling training data...')
# FIXME: pick_positive_name ignores those whose gold standard length is not one (multiple or nil)

'''
positives = [sp_training.pick_positive_name(config,corpus_train,concept,i) for i in range(len(corpus_train.names))]
positives = [*zip(positives,corpus_train.names)]

positives_dev_truncated = [sp_training.pick_positive_name(config,corpus_dev_truncated,concept,i) for i in range(len(corpus_dev_truncated.names))]
positives_dev_truncated = [*zip(positives_dev_truncated,corpus_dev_truncated.names)]

with open('gitig_positive_indices.pickle','wb') as f:
    pickle.dump([positives,positives_dev_truncated],f)
'''

positives_training, positives_dev_truncated = pickle.load(open('gitig_positive_indices.pickle','rb'))


# sampling
def examples(config, concept, positives, vocab, neg_count=config.getint('sample','neg_count')):
    """
    Builds positive and negative examples.
    """
    while True:
        for (chosen_idx, idces), span in positives:
        #for span, concept_id in zip(corpus_dev_truncated.names,corpus_dev_truncated.ids):
            e_tokens = nltk.word_tokenize(span)
            e_token_indices = [vocab.get(text.lower(),1) for text in e_tokens]
            
            if len(chosen_idx) ==1:
                # FIXME: only taking into account those that have exactly one gold concept
                c_tokens = nltk.word_tokenize(concept.names[chosen_idx[0]])
                c_token_indices = [vocab.get(text.lower(),1) for text in c_tokens]

            
            negative_concepts = [concept.names[i] for i in random.sample(list(set([*range(len(concept.names))])-set(idces)),neg_count)]
            negative_token_indices = [[vocab.get(t.lower(), 1) for t in nltk.word_tokenize(neg)] for neg in negative_concepts]
            
            entity_inputs = np.tile(pad_sequences([e_token_indices], padding='post', maxlen=config.getint('embedding','length')), (len(negative_concepts)+1, 1)) # Repeat the same entity for all concepts
            concept_inputs = pad_sequences([c_token_indices]+negative_token_indices, padding='post', maxlen=config.getint('embedding','length'))
            # concept_inputs = np.asarray([[concept_dict[cid]] for cid in [concept_id]+negative_concepts])
            # import pdb; pdb.set_trace()
            distances = [1] + [0]*len(negative_concepts)
            
            data = {
                'inp_mentions': entity_inputs,
                'inp_candidates': concept_inputs,
                'prediction_layer': np.asarray(distances),
            }
            
            yield data, data


from keras.models import Model
from keras.layers import Input, Embedding, Concatenate, Dense, Conv1D, GlobalMaxPooling1D, Flatten
from keras import layers

def evaluate_w_results(data_mentions, predictions, data_y, concept):
    '''
    Input:
    data_mentions: e.g. val_data.mentions, of the form [(start,end,untok_mention),(),...,()]
    predictions: [[prob],[prob],...,[prob]]
    data_y: e.g. val_data.y, of the form [[0],[1],...,[0]]
    '''
    assert len(predictions) == len(data_y)
    predictions = [item for sublist in predictions for item in sublist]
    correct = 0
    logger.warning('High chance of same prediction scores.')
    for start, end, untok_mention in data_mentions:
        index_prediction = np.argmax(predictions[start:end],axis=-1)
        if data_y[start:end][index_prediction] == 1:
        ##index_gold = np.argmax(data_y[start:end],axis=0)
        ##if index_prediction == index_gold:
            correct += 1
    else:
        print('Gold: {0}, Prediction: {1}'.format(untok_mention,concept.names[index_prediction]))
    total = len(data_mentions)
    accuracy = correct/total
    logger.info('Accuracy: {0}, Correct: {1}, Total: {2}'.format(accuracy,correct,total))
    return accuracy


def predict(config, concept, positives, vocab, entity_model, concept_model, original_model,val_data):
    entity_examples = examples(config, concept, positives, vocab, neg_count=0)

    c_token_indices = [[vocab.get(t.lower(), 1) for t in nltk.word_tokenize(neg)] for neg in concept.names]
    concept_examples = pad_sequences(c_token_indices, maxlen=config.getint('embedding','length'))
   
    entity_encodings = entity_model.predict_generator(entity_examples, steps=len(positives))    
    concept_encodings = concept_model.predict(concept_examples)

    ###################
    from sample import sped_up_format_x
    convoluted_input = sped_up_format_x(entity_encodings,concept_encodings)
    
    layers = ['v_sem','hidden_layer','prediction_layer']
    v_sem = original_model.get_layer(layers[0])
    d1 = original_model.get_layer(layers[1])
    d2 = original_model.get_layer(layers[2])

    entity_encodings = Input(shape=(convoluted_input[0].shape[1],),dtype='float32', name='entity_encodings')
    concept_encodings = Input(shape=(convoluted_input[1].shape[1],),dtype='float32', name='concept_encodings')
    sem = cnn.semantic_similarity_layer(weights = v_sem.get_weights())([entity_encodings,concept_encodings])
    concatenate_list = [entity_encodings,concept_encodings,sem]
    join_layer = Concatenate()(concatenate_list)
    hidden_layer = Dense(d1.units, activation=d1.activation,weights=d1.get_weights())(join_layer)
    prediction_layer = Dense(d2.units, activation=d2.activation,weights=d2.get_weights())(hidden_layer)

    model = Model(inputs=[entity_encodings,concept_encodings], outputs=prediction_layer)
    test_y = model.predict(convoluted_input)
    #evaluation_parameter = callback.evaluate(val_data.mentions, test_y, val_data.y)
    evaluate_w_results(val_data.mentions, test_y, val_data.y, concept)
    ###################
    # sims = cosine_similarity(entity_encodings, concept_encodings)
    
    # best_hits = np.argmax(sims, axis=-1)
    # predictions = [concept.ids[i] for i in best_hits]
    
    # return predictions
    return evaluation_parameter

def accuracy(labels,predictions):
    correct = 0
    for prediction, mention_gold in zip(predictions,labels):
        if prediction == mention_gold[0] and len(mention_gold)==1:
            correct += 1
    acc = correct/len(labels)
    logger.info('Accuracy:{0}'.format(acc)) 
    return acc

import random
from sklearn.metrics.pairwise import cosine_similarity

from model_tools import load_model
import cnn, model_tools, callback
from callback import EarlyStoppingRankingAccuracyGenerator


model, entity_model, concept_model = cnn.build_model_generator(config,vocabulary,pretrained)
train_examples = examples(config, concept, positives_training, vocabulary)
dev_examples = examples(config, concept, positives_dev_truncated, vocabulary)

accuracy_lst = []
for i in range(config.getint('training','epoch')):
    model.fit_generator(train_examples, steps_per_epoch=len(corpus_train.names), validation_data=dev_examples, validation_steps=len(corpus_dev_truncated.names), epochs=1)
    
    # preds = predict(config, concept, positives_dev_truncated, vocabulary, entity_model, concept_model, val_data)
    # accuracy_lst.append(accuracy(corpus_dev_truncated.ids,preds))

    acc = predict(config, concept, positives_dev_truncated, vocabulary, entity_model, concept_model,model, val_data_truncated)
    accuracy_lst.append(acc)

import pdb; pdb.set_trace()

# # shared_encoder_dot
# config['note']['note'] = 'shared encoder, cosine similarity'
# config['model']['path_model_whole'] = '/home/lhchan/disease_normalization/src/models/shared_encoder_dot.h5'

# evaluation_function = EarlyStoppingRankingAccuracySpedUpGiveModel(config,real_val_data,concept.padded,corpus_dev.padded,pretrained,cnn.forward_pass_speedup_shared_encoder_dot)
# #resample_function = Resample(tr_data,sampling,config,positives,concept,corpus_train.padded)
# cnn.print_input(tr_data)
# model = cnn.build_model_shared_encoder_dot(config,tr_data,vocabulary,pretrained)
# #model.load_weights('../gitig/model_whole.h5')

# for ep in range(50):
#     print('Epoch: {0}'.format(ep+1))
#     #from callback import EarlyStoppingRankingAccuracySpedUp, EarlyStoppingRankingAccuracySpedUpSharedEncoder
#     #evaluation_function_shared = EarlyStoppingRankingAccuracySpedUpSharedEncoder(config,real_val_data,concept.padded,corpus_dev.padded,pretrained)
#     if config['cnn']['loss'] == 'ranking_loss':
#         hist = model.fit(tr_data.x, tr_data.y, epochs=1, batch_size=config.getint('sample','neg_count')+1,callbacks=[evaluation_function],shuffle=0)
#         if ep!=100-1:
#             random.shuffle(positives)
#             tr_data = sampling(config,positives,concept,corpus_train.padded)
#     else:
#         hist = model.fit(tr_data.x, tr_data.y, epochs=1, batch_size=config.getint('sample','neg_count')+1,callbacks=[evaluation_function])
#         if ep!=100-1:
#             tr_data = sampling(config,positives,concept,corpus_train.padded)
# import pdb;pdb.set_trace()
# del model,hist,evaluation_function


# shared_encoder_dot_xDense
config['note']['note'] = 'shared encoder, cosine similarity, no dense layer'
config['cnn']['lr'] = '0.00001'
config['cnn']['loss'] == 'ranking_loss'
evaluation_function = EarlyStoppingRankingAccuracySpedUpGiveModel(config,real_val_data,concept.padded,corpus_dev.padded,pretrained,cnn.forward_pass_speedup_shared_encoder_dot_xDense)
cnn.print_input(tr_data)
model = cnn.build_model_shared_encoder_dot_xDense(config,tr_data,vocabulary,pretrained)

for ep in range(50):
    print('Epoch: {0}'.format(ep+1))
    if config['cnn']['loss'] == 'ranking_loss':
        hist = model.fit(tr_data.x, tr_data.y, epochs=1, batch_size=config.getint('sample','neg_count')+1,callbacks=[evaluation_function],shuffle=0)
        if ep!=100-1:
            random.shuffle(positives)
            tr_data = sampling(config,positives,concept,corpus_train.padded)
    else:
        hist = model.fit(tr_data.x, tr_data.y, epochs=1, batch_size=config.getint('sample','neg_count')+1,callbacks=[evaluation_function])
        if ep!=100-1:
            tr_data = sampling(config,positives,concept,corpus_train.padded)

exit()
# shared_encoder_dot_xDense, 1 kernel
config['note']['note'] = 'shared encoder (fully connected), cosine similarity, no dense layer'
config['cnn']['filters'] = '1'
config['cnn']['kernel_size'] = config['embedding']['length']
evaluation_function = EarlyStoppingRankingAccuracySpedUpGiveModel(config,real_val_data,concept.padded,corpus_dev.padded,pretrained,cnn.forward_pass_speedup_shared_encoder_dot_xDense)
cnn.print_input(tr_data)
model = cnn.build_model_shared_encoder_dot_xDense(config,tr_data,vocabulary,pretrained)

for ep in range(50):
    print('Epoch: {0}'.format(ep+1))
    if config['cnn']['loss'] == 'ranking_loss':
        hist = model.fit(tr_data.x, tr_data.y, epochs=1, batch_size=config.getint('sample','neg_count')+1,callbacks=[evaluation_function],shuffle=0)
        if ep!=100-1:
            random.shuffle(positives)
            tr_data = sampling(config,positives,concept,corpus_train.padded)
    else:
        hist = model.fit(tr_data.x, tr_data.y, epochs=1, batch_size=config.getint('sample','neg_count')+1,callbacks=[evaluation_function])
        if ep!=100-1:
            tr_data = sampling(config,positives,concept,corpus_train.padded)