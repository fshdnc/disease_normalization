'''Reorganized code, elmo without candidate generation'''

import logging
import logging.config

import configparser as cp
import args

import numpy as np
from keras.preprocessing.sequence import pad_sequences

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
vector_model, vocabulary, inversed_vocabulary = vectorizer.prepare_embedding_vocab('~/old-disease-normalization/data/embeddings/wvec_50_haodi-li-et-al.bin', binary = True, limit = 50000)
pretrained = vectorizer.load_pretrained_word_embeddings(vocabulary, vector_model)


# MEDIC dictionary
dictionary = load.Terminology()
# dictionary of entries, key = canonical id, value = named tuple in the form of
#   MEDIC_ENTRY(DiseaseID='MESH:D005671', DiseaseName='Fused Teeth',
#   AllDiseaseIDs=('MESH:D005671',), AllNames=('Fused Teeth', 'Teeth, Fused')
dictionary.loaded = load.load(config['terminology']['dict_file'],'MEDIC')


concept_ids = [] # list of all concept ids
concept_all_ids = [] # list of (lists of all concept ids with alt IDs)
concept_names = [] # list of all names, same length as concept_ids
concept_map = {} # names as keys, ids as concepts

for k in dictionary.loaded.keys(): # keys should be in congruent order
    c_id = dictionary.loaded[k].DiseaseID
    a_ids = dictionary.loaded[k].AllDiseaseIDs
    
    for n in dictionary.loaded[k].AllNames:
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
concept_vectorize = np.array([[vocabulary.get(text,1) for text in concept] for concept in concept_tokenize])
if config.getint('embedding','elmo'):
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
if config.getint('embedding','elmo'):
    concept.elmo = concept_elmo


# corpus
corpus_train = sample.NewDataSet('training corpus')
corpus_train.objects = load.load(config['corpus']['training_file'],'NCBI')

corpus_dev = sample.NewDataSet('dev corpus')
corpus_dev.objects = load.load(config['corpus']['development_file'],'NCBI')

for corpus in [corpus_train, corpus_dev]:
    mention_ids = [] # list of all ids (gold standard for each mention)
    mention_names = [] # list of all names
    mention_all = [] # list of tuples (mention_text,gold,context,(start,end,docid))

    #sth wrong here that sometimes throw an error
    #import pdb;pdb.set_trace()
    for abstract in corpus.objects:
        for section in abstract.sections: # title and abstract
            for mention in section.mentions:
                mention_ids.append(mention.id) # append list of ids, usually len(list)=1
                mention_names.append(mention.text)
                mention_all.append((mention.text,mention.id,section.text,(mention.start,mention.end,abstract.docid)))

    # tokenization & vectorization of mentions
    mention_tokenize = [nltk.word_tokenize(name) for name in mention_names]
    mention_vectorize = np.array([[vocabulary.get(text,1) for text in mention] for mention in mention_tokenize])
    if config.getint('embedding','elmo'):
        mention_elmo = elmo_default([mention_names])

    corpus.ids = mention_ids
    corpus.names = mention_names
    corpus.tokenize = mention_tokenize
    corpus.vectorize = mention_vectorize
    if config.getint('embedding','elmo'):
        corpus.elmo = mention_elmo


# padding
for corpus in [concept,corpus_train,corpus_dev]:
    logger.info('Padding {0}'.format(corpus.info))
    logger.info('Old shape: {0}'.format(corpus.vectorize.shape))
    corpus.padded = pad_sequences(corpus.vectorize, padding='post', maxlen=int(config['embedding']['length']))
    #format of corpus.padded: numpy, mentions, padded
    logger.info('New shape: {0}'.format(corpus.padded.shape))


# format data for cnn
try:
    import pickle
    [tr_data,val_data] = pickle.load(open('gitig_new_data_truncated.pickle','rb'))
    tr_data.y=np.array(tr_data.y)
    val_data.y=np.array(val_data.y)
    logger.info('Using saved data: {0}'.format('gitig_new_truncated.pickle'))
    #import pdb;pdb.set_trace()

except OSError:
    tr_data = sample.Data()
    val_data = sample.Data()
    for data, corpus in zip([tr_data, val_data],[corpus_train, corpus_dev]):
        data.x = sample.no_cangen_format_x(corpus.padded,concept.padded)
        data.mentions = sample.no_cangen_format_mentions(corpus.names,len(concept.names))
        data.y = [[1] if men[0] in can and len(men)==1 else [0] for men in corpus.ids for can in concept.all_ids]
        data.y = [item for sublist in data.y for item in sublist]
        assert len(data.x[0]) == len(data.y)
        
    # save the data for cnn since it takes forever to generate
    import pickle
    data = [tr_data,val_data]
    with open('gitig_new_data.pickle','wb') as f:
        pickle.dump(data,f,protocol=4)
    logger.info('Mentions and concepts saved.')


# cnn
if not int(config['model']['use_saved_model']):    # train new model
    import cnn, model_tools
    from callback import EarlyStoppingRankingAccuracy
    evaluation_function = EarlyStoppingRankingAccuracy(config,val_data)
    cnn.print_input(tr_data)
    model = cnn.build_model(config,tr_data,vocabulary,pretrained)
    
    if int(config['settings']['imp_tr']):
    #importance sampling
        from importance_sampling.training import ImportanceTraining
        logger.warning('Using truncated data!')
        fake_data_x = [a[:1000000]for a in tr_data.x]
        hist = ImportanceTraining(model).fit(fake_data_x, tr_data.y[:1000000], epochs=int(config['training']['epoch']), batch_size=100, callbacks=[evaluation_function])
    else:
        #logger.warning('Using truncated data!')
        #fake_data_x = [a[:10000]for a in tr_data.x]
        #hist = model.fit(fake_data_x, tr_data.y[:10000], epochs=int(config['training']['epoch']), batch_size=100, callbacks=[evaluation_function])
        hist = model.fit(tr_data.x, tr_data.y, epochs=int(config['training']['epoch']), batch_size=100, callbacks=[evaluation_function])
    
    #hist = model.fit(tr_data.x, tr_data.y, epochs=int(config['training']['epoch']), batch_size=100, callbacks=[evaluation_function])
    # WARNING (theano.tensor.blas): We did not found a dynamic library into the library_dir of the library we use for blas. If you use ATLAS, make sure to compile it with dynamics library.
    logger.info('Saving newly trained model...')
    model_tools.save_model(model,config['model']['path_model_architecture'],config['model']['path_model_weights'])
else:
    from cnn import semantic_similarity_layer
    import cnn, model_tools
    model = model_tools.load_model(config['model']['path_model_architecture'],config['model']['path_model_weights'],{'semantic_similarity_layer': semantic_similarity_layer})
    model.compile(optimizer='adadelta',loss='binary_crossentropy')