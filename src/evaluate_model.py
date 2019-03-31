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
config['cnn']['filters'] = '50'
config['cnn']['optimizer'] = 'adam'
config['cnn']['lr'] = '0.0001'
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


#[real_val_data,concept_order] = pickle.load(open('gitig_real_val_data.pickle','rb'))
#real_val_data.y=np.array(real_val_data.y)

#truncated real eval data
[real_val_data,concept_order,mention_padded] = pickle.load(open('gitig_real_val_data_truncated.pickle','rb'))
real_val_data.y=np.array(real_val_data.y)


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

concept = concept_obj(config,dictionary,order=concept_order)


# corpus
corpus_dev = sample.NewDataSet('dev corpus')
corpus_dev.objects = load.load(config['corpus']['development_file'],'NCBI')

for corpus in [corpus_dev]:
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

for corpus in [corpus_dev]:
    logger.info('Padding {0}'.format(corpus.info))
    logger.info('Old shape: {0}'.format(corpus.vectorize.shape))
    corpus.padded = pad_sequences(corpus.vectorize, padding='post', maxlen=int(config['embedding']['length']))
    #format of corpus.padded: numpy, mentions, padded
    logger.info('New shape: {0}'.format(corpus.padded.shape))


# load the model to be evaluated
from cnn import semantic_similarity_layer
#from keras.models import load_model
#model = load_model('models/20190324-220005.h5',custom_objects={'semantic_similarity_layer': semantic_similarity_layer})
from model_tools import load_model
import cnn, model_tools, callback
from keras.models import Model
from keras.layers import Input, Embedding, Concatenate, Dense, Conv1D, GlobalMaxPooling1D, Flatten
from keras import layers
#model = model_tools.load_model('models/20190329-142605.json','models/synpair_continued_62.h5',{'semantic_similarity_layer': semantic_similarity_layer})
model = model_tools.load_model('models/20190324-220005.json','models/20190324-220005.h5',{'semantic_similarity_layer': semantic_similarity_layer})

model.summary()

# def forward_pass_speedup_shared_encoder(model,corpus_padded,concept_padded,pretrained):
#     '''
#     Model to speed up forward pass, used in callback for evaluation
#     '''
#     model_mention = _forward_pass_speedup_conv(model,['inp_mentions','embedding_1','conv1d_1','global_max_pooling1d_1'],pretrained)
#     mentions = model_mention.predict(corpus_padded) # (787, 50)
#     model_candidate = _forward_pass_speedup_conv(model,['inp_candidates','embedding_1','conv1d_1','global_max_pooling1d_2'],pretrained)
#     candidates = model_candidate.predict(concept_padded) # (67782,50)
#     logger.info('Formatting pooled mentions and candidates...')
#     # from sample import no_cangen_format_x
#     from sample import sped_up_format_x
#     convoluted_input = sped_up_format_x(mentions,candidates)
#     model_sem = _forward_pass_speedup_sem(model,convoluted_input)
#     return convoluted_input, model_sem

# def _forward_pass_speedup_sem(original_model,convoluted_x):
#     layers = ['semantic_similarity_layer_1','dense_1','dense_2']
#     v_sem = original_model.get_layer(layers[0])
#     d1 = original_model.get_layer(layers[1])
#     d2 = original_model.get_layer(layers[2])

#     pooled_mentions = Input(shape=(convoluted_x[0].shape[1],),dtype='float32', name='pooled_mentions')
#     pooled_candidates = Input(shape=(convoluted_x[1].shape[1],),dtype='float32', name='pooled_candidates')
#     sem = semantic_similarity_layer(weights = v_sem.get_weights())([pooled_mentions,pooled_candidates])
#     concatenate_list = [pooled_mentions,pooled_candidates,sem]
#     join_layer = Concatenate()(concatenate_list)
#     hidden_layer = Dense(d1.units, activation=d1.activation,weights=d1.get_weights())(join_layer)
#     prediction_layer = Dense(d2.units, activation=d2.activation,weights=d2.get_weights())(hidden_layer)
    
#     input_list = [pooled_mentions, pooled_candidates]
#     model_part = Model(inputs=input_list, outputs=prediction_layer)
#     return model_part

# def _forward_pass_speedup_conv(original_model,layers,pretrained):
#     '''
#     Input:
#     original_model
#     layers: list of layer names, one of the two
#         ['inp_mentions','embedding_1','conv1d_1','global_max_pooling1d_1']
#         ['inp_candidates','embedding_1','conv1d_2','global_max_pooling1d_2']
#     '''
#     terms = original_model.get_layer(layers[0])
#     emb = original_model.get_layer(layers[1])
#     conv = original_model.get_layer(layers[2])

#     new_input_terms = Input(shape=(terms.input_shape[1],),dtype='int32', name='new_input_terms')
#     new_emb = Embedding(emb.input_dim, emb.output_dim, mask_zero=False, trainable=False, weights=[pretrained])
#     encoded = new_emb(new_input_terms)
#     new_conv = Conv1D(filters=conv.filters,kernel_size=conv.kernel_size[0],activation=conv.activation,weights=conv.get_weights())(encoded)
#     gl_max_p = GlobalMaxPooling1D()(new_conv)

#     model_part = Model(inputs=new_input_terms, outputs=gl_max_p)
#     return model_part

def _forward_pass_speedup_conv(original_model,layerss,pretrained):
    '''
    Input:
    original_model
    layerss: list of layer names, one of the two
        ['inp_mentions','embedding_1','drop',conv1d_1','global_max_pooling1d_1']
        ['inp_candidates','embedding_1','drop','conv1d_2','global_max_pooling1d_2']
    '''
    terms = original_model.get_layer(layerss[0])
    emb = original_model.get_layer(layerss[1])
    #drop = original_model.get_layer(layerss[2])
    conv = original_model.get_layer(layerss[2])

    new_input_terms = Input(shape=(terms.input_shape[1],),dtype='int32', name='new_input_terms')
    new_emb = Embedding(emb.input_dim, emb.output_dim, mask_zero=False, trainable=False, weights=[pretrained])
    new_drop = layers.Dropout(0)
    encoded = new_drop(new_emb(new_input_terms))
    new_conv = Conv1D(filters=conv.filters,kernel_size=conv.kernel_size[0],activation=conv.activation,weights=conv.get_weights())(encoded)
    gl_max_p = GlobalMaxPooling1D()(new_conv)

    model_part = Model(inputs=new_input_terms, outputs=gl_max_p)
    return model_part

def _forward_pass_speedup_sem(original_model,convoluted_x):
    layers = ['semantic_similarity_layer_1','dense_1','dense_2']
    v_sem = original_model.get_layer(layers[0])
    d1 = original_model.get_layer(layers[1])
    d2 = original_model.get_layer(layers[2])

    pooled_mentions = Input(shape=(convoluted_x[0].shape[1],),dtype='float32', name='pooled_mentions')
    pooled_candidates = Input(shape=(convoluted_x[1].shape[1],),dtype='float32', name='pooled_candidates')
    sem = semantic_similarity_layer(weights = v_sem.get_weights())([pooled_mentions,pooled_candidates])
    concatenate_list = [pooled_mentions,pooled_candidates,sem]
    join_layer = Concatenate()(concatenate_list)
    hidden_layer = Dense(d1.units, activation=d1.activation,weights=d1.get_weights())(join_layer)
    prediction_layer = Dense(d2.units, activation=d2.activation,weights=d2.get_weights())(hidden_layer)
    
    input_list = [pooled_mentions, pooled_candidates]
    model_part = Model(inputs=input_list, outputs=prediction_layer)
    return model_part

def forward_pass_speedup_shared_encoder(model,corpus_padded,concept_padded,pretrained):
    '''
    Model to speed up forward pass, used in callback for evaluation
    '''
    model_mention = _forward_pass_speedup_conv(model,['inp_mentions','embedding_1','conv1d_1','global_max_pooling1d_1'],pretrained)
    mentions = model_mention.predict(corpus_padded) # (787, 50)
    model_candidate = _forward_pass_speedup_conv(model,['inp_candidates','embedding_1','conv1d_1','global_max_pooling1d_2'],pretrained)
    candidates = model_candidate.predict(concept_padded) # (67782,50)
    logger.info('Formatting pooled mentions and candidates...')
    # from sample import no_cangen_format_x
    from sample import sped_up_format_x
    convoluted_input = sped_up_format_x(mentions,candidates)
    model_sem = _forward_pass_speedup_sem(model,convoluted_input)
    return convoluted_input, model_sem

conv_x, sped_up_model = forward_pass_speedup_shared_encoder(model,mention_padded,concept.padded,pretrained)                                                                  
tr_predictions = sped_up_model.predict(conv_x)
acc = callback.evaluate(real_val_data.mentions,tr_predictions,real_val_data.y)
logger.info('Accuracy for the pretrained model on validation set:{0}'.format(acc))