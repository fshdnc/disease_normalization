'''Pretraining
- use all concepts and synonyms
-exact match pairs and synonym pairs'''

import logging
import logging.config

import configparser as cp
import args

import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pickle
import random
import nltk

import vectorizer
import load
import sample

#configurations
config = cp.ConfigParser(strict=False)
config.read('defaults.cfg')
#################################################
config['embedding']['emb_file'] = '/home/lhchan/disease_normalization/data/pubmed2018_w2v_400D/pubmed2018_w2v_400D.bin'
#'/home/lenz/disease-normalization/data/embeddings/wvec_50_haodi-li-et-al.bin'
config['cnn']['filters'] = '50'
config['cnn']['optimizer'] = 'adam'
config['cnn']['lr'] = '0.00001'
config['cnn']['loss'] = 'binary_crossentropy'
config['cnn']['dropout'] = '0'
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


def generate_synonym_pairs(dictionary, order=None):
    concept_synonyms = []

    if order:
        use = order
        logger.info('Re-initialing concept object.')
    else:
        use = dictionary.loaded.keys()

    for k in use:
        concept_synonyms.append(dictionary.loaded[k].AllNames)

    synonym_pairs = []
    for concept in concept_synonyms:
        for i,name in enumerate(concept):
            for j in range(len(concept)-i):
                synonym_pairs.append((name,concept[j]))

    return synonym_pairs

# get the real validation data
'''
[real_tr_data,real_val_data,concept_order] = pickle.load(open('gitig_new_data.pickle','rb'))
real_tr_data.y=np.array(real_tr_data.y)
real_val_data.y=np.array(real_val_data.y)
'''
logger.info('Using truncated development corpus for evaluation.')
corpus_dev = sample.NewDataSet('dev corpus')
[real_val_data,concept_order,corpus_dev.padded] = pickle.load(open('gitig_real_val_data_truncated.pickle','rb'))
real_val_data.y=np.array(real_val_data.y)

# reload the concept dict so that it is in the order when the data for predicion is created
concept = concept_obj(config,dictionary,order=concept_order)
'''
synonym_pairs = generate_synonym_pairs(dictionary,order=concept_order)
questions = [question for question, answer in synonym_pairs]
answers = [answer for question, answer in synonym_pairs]
# FIXME: there may be positives as well
# negatives = random.choices(concept.names,k=len(questions)) # this only works for python 3.6 +
negatives = [random.choice(concept.names) for i in range(len(questions))]


cutoff = len(synonym_pairs)*2 - len(synonym_pairs)*2//10
collection = []
for question, positive, negative in zip(questions,answers,negatives):
    collection.extend([(question,positive,1),(question,negative,0)])
random.shuffle(collection)
tr_collection = collection[:cutoff]
val_collection = collection[cutoff:]
tr_data = sample.Data()
val_data = sample.Data()


for sat, data in zip([tr_collection,val_collection],[tr_data,val_data]):
    x0 = []
    x1 = []
    y = []
    for q,a,l in sat:
        x0.append([vocabulary.get(tok.lower(),1) for tok in nltk.word_tokenize(q)])
        x1.append([vocabulary.get(tok.lower(),1) for tok in nltk.word_tokenize(a)])
        y.append(l)
    x0 = pad_sequences(np.array(x0), padding='post', maxlen=int(config['embedding']['length']))
    x1 = pad_sequences(np.array(x1), padding='post', maxlen=int(config['embedding']['length']))
    data.x = [x0,x1]
    data.y = np.array(y)


# cnn
# pre-train model
from callback import EarlyStoppingRankingAccuracy, EarlyStoppingRankingAccuracySpedUpSharedEncoder
import cnn, model_tools
cnn.print_input(tr_data)
# model = cnn.build_model_shared_encoder(config,tr_data,vocabulary,pretrained)

# pretraining with shared encoders
evaluation_function_truncated_dev = EarlyStoppingRankingAccuracySpedUpSharedEncoder(config,real_val_data,concept.padded,corpus_dev.padded,pretrained)

#model_shared_encoder = cnn.build_model_shared_encoder(config,tr_data,vocabulary,pretrained)
#model_shared_encoder.summary()


from cnn import semantic_similarity_layer
model_shared_encoder = model_tools.load_model('models/20190329-142605.json','models/20190329-142605.h5',{'semantic_similarity_layer': semantic_similarity_layer})
model_shared_encoder.compile(optimizer='adam',loss='binary_crossentropy')

hist_shared = model_shared_encoder.fit(tr_data.x, tr_data.y, epochs=100, batch_size=100,callbacks=[evaluation_function_truncated_dev])
#import pdb; pdb.set_trace()
logger.info('Training loss (shared encoder):', hist_shared.history['loss'])
logger.info('Validation accuracy (shared encoder):', evaluation_function_shared.accuracy)
import pdb; pdb.set_trace()
'''


# continue training with real data
positives = pickle.load(open('gitig_positive_indices.pickle','rb'))
#positives = pickle.load(open('gitig_positive_indices_all.pickle','rb'))

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

import sp_training
def sampling(conf,positives,concept,corpus_train_padded):
    logger.info('Sampling training examples...')
    sampled = [sp_training.sample_for_individual_mention(pos,len(concept.names),config.getint('sample','neg_count')) for pos,men in positives]
    name_order = [men for pos,men in positives]
    tr_data = sample.Data()
    tr_data.mentions = sample.sample_format_mentions(sampled,name_order)
    tr_data.x = sample.sample_format_x(sampled,corpus_train.padded,concept.padded,tr_data.mentions)
    tr_data.y = sample.sample_format_y(sampled)
    assert len(tr_data.x[0]) == len(tr_data.y)
    return tr_data
tr_data = sampling(config,positives,concept,corpus_train.padded)

from callback import EarlyStoppingRankingAccuracySpedUpSharedEncoder
from keras.callbacks import CSVLogger

import cnn, model_tools
cnn.print_input(tr_data)

evaluation_function_truncated_dev = EarlyStoppingRankingAccuracySpedUpSharedEncoder(config,real_val_data,concept.padded,corpus_dev.padded,pretrained)
from datetime import datetime

from cnn import semantic_similarity_layer
model_shared_encoder = model_tools.load_model('models/20190329-142605.json','models/pretrained_synpair.h5',{'semantic_similarity_layer': semantic_similarity_layer})
from keras import optimizers
adam = optimizers.Adam(lr=0.00001, epsilon=None, decay=0.0)
model_shared_encoder.compile(optimizer=adam,loss='binary_crossentropy')


for ep in range(100):
    print('Epoch: {0}'.format(ep+1))
    hist_shared = model_shared_encoder.fit(tr_data.x, tr_data.y, epochs=1, batch_size=100,callbacks=[evaluation_function_truncated_dev])
    tr_data = sampling(config,positives,concept,corpus_train.padded)
import pdb; pdb.set_trace()