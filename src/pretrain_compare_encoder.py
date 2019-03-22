'''Pretraining
- using the real validation set
refer to jupyter notebook pretrain_compair_encoder.ipynb'''

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
    # concept_ids = [] # list of all concept ids
    # concept_all_ids = [] # list of (lists of all concept ids with alt IDs)
    concept_names = [] # list of all names, same length as concept_ids
    # concept_map = {} # names as keys, ids as concepts

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
        # c_id = dictionary.loaded[k].DiseaseID
        # a_ids = dictionary.loaded[k].AllDiseaseIDs
        for n in dictionary.loaded[k].AllNames:
            concept_names.append(n)

    # tokenization & vectorization of dictionary terms
    import nltk
    concept_tokenize = [nltk.word_tokenize(name) for name in concept_names] # list of list of tokenized names
    concept_vectorize = np.array([[vocabulary.get(text.lower(),1) for text in concept] for concept in concept_tokenize])


    # save the stuff to object
    concept = sample.NewDataSet('concepts')
    #concept.ids = concept_ids
    #concept.all_ids = concept_all_ids
    concept.names = concept_names
    #concept.map = concept_map
    concept.tokenize = concept_tokenize
    concept.vectorize = concept_vectorize
    for corpus in [concept]:
        logger.info('Padding {0}'.format(corpus.info))
        logger.info('Old shape: {0}'.format(corpus.vectorize.shape))
        corpus.padded = pad_sequences(corpus.vectorize, padding='post', maxlen=int(config['embedding']['length']))
        #format of corpus.padded: numpy, mentions, padded
        logger.info('New shape: {0}'.format(corpus.padded.shape))

    return concept


# get the real validation data
'''
[real_tr_data,real_val_data,concept_order] = pickle.load(open('gitig_new_data.pickle','rb'))
real_tr_data.y=np.array(real_tr_data.y)
real_val_data.y=np.array(real_val_data.y)
'''
[real_val_data,concept_order] = pickle.load(open('gitig_real_val_data.pickle','rb'))
real_val_data.y=np.array(real_val_data.y)

# reload the concept dict so that it is in the order when the data for predicion is created
concept = concept_obj(config,dictionary,order=concept_order)


# corpus
corpus_train = sample.NewDataSet('training corpus')
corpus_train.objects = load.load(config['corpus']['training_file'],'NCBI')

corpus_dev = sample.NewDataSet('dev corpus')
corpus_dev.objects = load.load(config['corpus']['development_file'],'NCBI')

for corpus in [corpus_train, corpus_dev]:
    # mention_ids = [] # list of all ids (gold standard for each mention)
    mention_names = [] # list of all names
    # mention_all = [] # list of tuples (mention_text,gold,context,(start,end,docid))

    #sth wrong here that sometimes throw an error
    #import pdb;pdb.set_trace()
    for abstract in corpus.objects:
        for section in abstract.sections: # title and abstract
            for mention in section.mentions:
                # nor_ids = [sample._nor_id(one_id) for one_id in mention.id]
                # mention_ids.append(nor_ids) # append list of ids, usually len(list)=1
                mention_names.append(mention.text)
                # mention_all.append((mention.text,nor_ids,section.text,(mention.start,mention.end,abstract.docid)))

    # tokenization & vectorization of mentions
    import nltk
    mention_tokenize = [nltk.word_tokenize(name) for name in mention_names]
    mention_vectorize = np.array([[vocabulary.get(text.lower(),1) for text in mention] for mention in mention_tokenize])

    # corpus.ids = mention_ids
    corpus.names = mention_names
    # corpus.all = mention_all
    corpus.tokenize = mention_tokenize
    corpus.vectorize = mention_vectorize

for corpus in [corpus_train,corpus_dev]:
    logger.info('Padding {0}'.format(corpus.info))
    logger.info('Old shape: {0}'.format(corpus.vectorize.shape))
    corpus.padded = pad_sequences(corpus.vectorize, padding='post', maxlen=int(config['embedding']['length']))
    #format of corpus.padded: numpy, mentions, padded
    logger.info('New shape: {0}'.format(corpus.padded.shape))


collection_names = concept.names + corpus_train.names
collection_tokenize = concept.tokenize + corpus_train.tokenize
collection = np.concatenate((concept.padded, corpus_train.padded),axis=0)
# split the training and validation set
# cutoff = len(collection) - len(collection)//10


# questions, answers, labels
questions = [term for term in collection for i in range(2)]
answers = []
labels = []
mentions = []

import random

for i, c in enumerate(collection):
    order = random.randint(0,1)
    j = i-1000
    try:
        # test that the vectorized 'random' sample is not the same as the question
        identical = c==collection[j]
        assert not identical.all()
    except AssertionError:
        identical = c==collection[j]
        while identical.all():
            j = j -10
            identical = c==collection[j]
    if order:
        answers.append(c)
        answers.append(collection[j])
        labels.extend([1,0])
    else:
        answers.append(collection[j])
        answers.append(c)
        labels.extend([0,1])


# using all synthetic data for training, real development data for validation
tr_data = sample.Data()
tr_data.x = [np.array(questions),np.array(answers)]
tr_data.y = np.array(labels)
tr_data.mentions=[]
for i, c in enumerate(collection):
    tr_data.mentions.append((i*2,i*2+2,collection_names[i]))


'''
tr_data = sample.Data()
val_data = sample.Data()
tr_data.x = [np.array(questions[:cutoff*2]),np.array(answers[:cutoff*2])]
val_data.x = [np.array(questions[cutoff*2:]),np.array(answers[cutoff*2:])]
tr_data.y = np.array(labels[:cutoff*2])
val_data.y = np.array(labels[cutoff*2:])
tr_data.mentions=[]
val_data.mentions=[]
for i, c in enumerate(collection[:cutoff]):
    tr_data.mentions.append((i*2,i*2+2,collection_names[i]))
for i, c in enumerate(collection[cutoff:]):
    val_data.mentions.append((i*2,i*2+2,collection_names[i]))
# assert len(val_data.mentions)==len(val_data.y)/2
'''

# cnn, filter = 70
import cnn, model_tools
cnn.print_input(tr_data)
from callback import EarlyStoppingRankingAccuracySpedUp, EarlyStoppingRankingAccuracySpedUpSharedEncoder
from model_tools import save_model

'''
# pretraining with shared encoders
evaluation_function_shared = EarlyStoppingRankingAccuracySpedUpSharedEncoder(config,real_val_data,concept.padded,corpus_dev.padded,pretrained)
model_shared_encoder = cnn.build_model_shared_encoder(config,tr_data,vocabulary,pretrained)
model_shared_encoder.summary()
hist_shared = model_shared_encoder.fit(tr_data.x, tr_data.y, epochs=15, batch_size=100,callbacks=[evaluation_function_shared])
logger.info('Training loss (shared encoder):', hist_shared.history['loss'])
logger.info('Validation accuracy (shared encoder):', evaluation_function_shared.accuracy)
m = 'shared_encoder'
model = model_shared_encoder

# pretraining with separate encoders
evaluation_function_separate = EarlyStoppingRankingAccuracySpedUp(config,real_val_data,concept.padded,corpus_dev.padded,pretrained)
model_separate_encoder = cnn.build_model(config,tr_data,vocabulary,pretrained)
model_separate_encoder.summary()
hist_separate = model_separate_encoder.fit(tr_data.x, tr_data.y, epochs=15, batch_size=100,callbacks=[evaluation_function_separate])
logger.info('Training loss (separate encoder):', hist_separate.history['loss'])
logger.info('Validation accuracy (separate encoder):', evaluation_function_separate.accuracy)
m = 'separate_encoder'
model = model_separate_encoder
'''

# pretraining with shared encoders, no dense layer
from callback import EarlyStoppingRankingAccuracySpedUpGiveModel
evaluation_function_shared_xDense = EarlyStoppingRankingAccuracySpedUpGiveModel(config,real_val_data,concept.padded,corpus_dev.padded,pretrained,cnn.forward_pass_speedup_shared_encoder_xDense)
model_shared_encoder_xDense = cnn.build_model_shared_encoder_xDense(config,tr_data,vocabulary,pretrained)
model_shared_encoder_xDense.summary()
hist_shared_xDense = model_shared_encoder_xDense.fit(tr_data.x, tr_data.y, epochs=15, batch_size=100,callbacks=[evaluation_function_shared_xDense])
logger.info('Training loss (shared encoder, no dense layer):', hist_shared_xDense.history['loss'])
logger.info('Validation accuracy (shared encoder, no dense layer):', evaluation_function_shared_xDense.accuracy)
m = 'shared_encoder_xDense'
model = model_shared_encoder_xDense


# Save the model
path = 'model/'
parameters = '_d400_f70'
model_name = ''.join([path, m, parameters, '.json'])
weights_name = ''.join([path, m, parameters, '.h5'])
save_model(model,model_name,weights_name)
logger.info('Model saved to {0}.'.format(model_name))