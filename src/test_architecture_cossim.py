'''
use dot product instead of similarity layer
'''


import logging
import logging.config

import configparser as cp
import args

import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pickle

import vectorizer
import load
import sample

#configurations
config = cp.ConfigParser(strict=False)
config.read('defaults.cfg')
#################################################
config['embedding']['emb_file'] = '/home/lhchan/disease_normalization/data/pubmed2018_w2v_400D/pubmed2018_w2v_400D.bin'
config['cnn']['filters'] = '50'
config['cnn']['optimizer'] = 'adam'
config['cnn']['lr'] = '0.0001'
config['cnn']['loss'] = 'binary_crossentropy'
config['cnn']['dropout'] = '0.5'
config['embedding']['length'] = '20'
config['embedding']['limit'] = '1000000'
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
vector_model, vocabulary, inversed_vocabulary = vectorizer.prepare_embedding_vocab('/home/lhchan/disease_normalization/data/pubmed2018_w2v_400D/pubmed2018_w2v_400D.bin', binary = True, limit = 1000000)
# vector_model, vocabulary, inversed_vocabulary = vectorizer.prepare_embedding_vocab('/home/lenz/disease-normalization/data/embeddings/wvec_50_haodi-li-et-al.bin', binary = True, limit = 1000000)
pretrained = vectorizer.load_pretrained_word_embeddings(vocabulary, vector_model)

# MEDIC dictionary
dictionary = load.Terminology()
# dictionary of entries, key = canonical id, value = named tuple in the form of
#   MEDIC_ENTRY(DiseaseID='MESH:D005671', DiseaseName='Fused Teeth',
#   AllDiseaseIDs=('MESH:D005671',), AllNames=('Fused Teeth', 'Teeth, Fused')
dictionary.loaded = load.load(config['terminology']['dict_file'],'MEDIC')


def concept_obj(conf,dictionary,order=None):
    concept_ids = [] # list of all concept ids
    concept_all_ids = [] # list of (lists of all concept ids with alt IDs)
    concept_names = [] # list of all names, same length as concept_ids
    concept_map = {} # names as keys, ids as concepts

    if order:
        use = order
        logger.info('Re-initialing concept object.')
    else:
        use = dictionary.loaded.keys()

    for k in use:
    # keys not in congruent order! To make them congruent:
    # k,v = zip(*dictionary.loaded.items())
    # k = list(k)
    # k.sort()
        c_id = dictionary.loaded[k].DiseaseID
        a_ids = dictionary.loaded[k].AllDiseaseIDs
        
        if int(conf['settings']['all_names']):
            for n in dictionary.loaded[k].AllNames:
                concept_ids.append(c_id)
                concept_all_ids.append(a_ids)
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


    # tokenization & vectorization of dictionary terms
    import nltk
    concept_tokenize = [nltk.word_tokenize(name) for name in concept_names] # list of list of tokenized names
    concept_vectorize = np.array([[vocabulary.get(text.lower(),1) for text in concept] for concept in concept_tokenize])
    if conf.getint('embedding','elmo'):
        from vectorizer_elmo import elmo_default
        concept_elmo = elmo_default([concept_names])

    # save the stuff to object
    concept = sample.NewDataSet('concepts')
    concept.ids = concept_ids
    concept.all_ids = concept_all_ids
    concept.names = concept_names
    concept.map = concept_map
    concept.tokenize = concept_tokenize
    concept.vectorize = concept_vectorize
    if conf.getint('embedding','elmo'):
        concept.elmo = concept_elmo

    logger.info('Padding {0}'.format(concept.info))
    logger.info('Old shape: {0}'.format(concept.vectorize.shape))
    concept.padded = pad_sequences(concept.vectorize, padding='post', maxlen=int(config['embedding']['length']))
    #format of corpus.padded: numpy, mentions, padded
    logger.info('New shape: {0}'.format(concept.padded.shape))

    return concept


# Get the concept order
corpus_dev = sample.NewDataSet('dev corpus')
[real_val_data,concept_order,corpus_dev.padded] = pickle.load(open('gitig_real_val_data_truncated.pickle','rb'))
#del real_val_data, corpus_dev_padded
concept = concept_obj(config,dictionary,order=concept_order)


# corpus
corpus_train = sample.NewDataSet('training corpus')
corpus_train.objects = load.load(config['corpus']['training_file'],'NCBI')

for corpus in [corpus_train]:
    mention_ids = [] # list of all ids (gold standard for each mention)
    mention_names = [] # list of all names
    mention_all = [] # list of tuples (mention_text,gold,context,(start,end,docid))

    #sth wrong here that sometimes throw an error
    #import pdb;pdb.set_trace()
    for abstract in corpus.objects:
        for section in abstract.sections: # title and abstract
            for mention in section.mentions:
                nor_ids = [sample._nor_id(one_id) for one_id in mention.id]
                mention_ids.append(nor_ids) # append list of ids, usually len(list)=1
                mention_names.append(mention.text)
                mention_all.append((mention.text,nor_ids,section.text,(mention.start,mention.end,abstract.docid)))

    # tokenization & vectorization of mentions
    import nltk
    mention_tokenize = [nltk.word_tokenize(name) for name in mention_names]
    mention_vectorize = np.array([[vocabulary.get(text.lower(),1) for text in mention] for mention in mention_tokenize])
    if config.getint('embedding','elmo'):
        mention_elmo = elmo_default([mention_names])

    corpus.ids = mention_ids
    corpus.names = mention_names
    corpus.all = mention_all
    corpus.tokenize = mention_tokenize
    corpus.vectorize = mention_vectorize
    if config.getint('embedding','elmo'):
        from vectorizer_elmo import elmo_default
        corpus.elmo = mention_elmo

# padding
for corpus in [corpus_train]:
    logger.info('Padding {0}'.format(corpus.info))
    logger.info('Old shape: {0}'.format(corpus.vectorize.shape))
    corpus.padded = pad_sequences(corpus.vectorize, padding='post', maxlen=int(config['embedding']['length']))
    #format of corpus.padded: numpy, mentions, padded
    logger.info('New shape: {0}'.format(corpus.padded.shape))


# sample concepts for each mention
import sp_training, sample
logger.info('Sampling training data...')
'''
positives = [sp_training.pick_positive_name(config,corpus_train,concept,i) for i in range(len(corpus_train.names))]
positives = [*zip(positives,corpus_train.names)]
with open('gitig_positive_indices.pickle','wb') as f:
    pickle.dump(positives,f)
'''
positives = pickle.load(open('gitig_positive_indices.pickle','rb'))

def sampling(conf,positives,concept,corpus_train_padded):
    logger.info('Resampling training data...')
    sampled = [sp_training.sample_for_individual_mention(pos,len(concept.names),config.getint('sample','neg_count')) for pos,men in positives]
    name_order = [men for pos,men in positives]
    tr_data = sample.Data()
    tr_data.mentions = sample.sample_format_mentions(sampled,name_order)
    tr_data.x = sample.sample_format_x(sampled,corpus_train.padded,concept.padded,tr_data.mentions)
    tr_data.y = sample.sample_format_y(sampled)
    assert len(tr_data.x[0]) == len(tr_data.y)
    return tr_data
tr_data = sampling(config,positives,concept,corpus_train.padded)

from model_tools import load_model
import cnn, model_tools, callback
import random
from callback import EarlyStoppingRankingAccuracySpedUpGiveModel


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