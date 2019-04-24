'''
Test different architectures: starting from shared encoder with dense layer
    1. shared encoder (CNN), semantic similarity layer, w/ dense layer
    2. shared encoder (CNN), semantic similarity layer, no dense layer
    3. shared encoder (CNN), dot product, w/ dense layer
    4. shared encoder (CNN), dot product, no dense layer
    5. tree lstm?
    6. 
first argv: 'separate', 'shared'
second argv: 'full', 'ablation'
'''
raise NotImplementedError('add synpair pre-training')

import sys
assert sys.argv[1] == 'sem_matrix' or sys.argv[1] == 'cosine_sim' or sys.argv[1] == 'no_sim'
assert sys.argv[2] == 'full' or sys.argv[2] == 'ablation'
sysargv1 = sys.argv[1]
sysargv2 = sys.argv[2]

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
dynamic_defaults = {'timestamp': time.strftime('%Y%m%d-%H%M%S')}
config = cp.ConfigParser(defaults=dynamic_defaults,interpolation=cp.ExtendedInterpolation(),strict=False)
try:
    directory = os.path.join(os.path.abspath(os.path.dirname(__file__)))
    config.read(os.path.join(directory, 'defaults.cfg'))
except NameError:
    directory = '/home/lhchan/disease_normalization/src'
    config.read(os.path.join(directory, 'defaults.cfg'))
#################################################
config['embedding']['emb_file'] = os.path.join(directory, '../../disease_normalization/data/pubmed2018_w2v_200D/pubmed2018_w2v_200D.bin')
config['terminology']['dict_file'] = os.path.join(directory, '../../old-disease-normalization/data/ncbi-disease/CTD_diseases.tsv')
config['corpus']['training_file'] = os.path.join(directory,'../../old-disease-normalization/data/ncbi-disease/NCBItrainset_corpus.txt')
config['corpus']['development_file'] = os.path.join(directory,'../../old-disease-normalization/data/ncbi-disease/NCBIdevelopset_corpus.txt')
config['settings']['history'] = os.path.join(directory, '../gitig/log/')
config['cnn']['filters'] = '50'
config['cnn']['optimizer'] = 'adam'
config['cnn']['lr'] = '0.00005'
config['cnn']['loss'] = 'binary_crossentropy'
config['cnn']['dropout'] = '0.5'
config['embedding']['length'] = '10'
config['embedding']['limit'] = '1000000'
config['note']['note'] = 'final experiment, testing architectures, architectue:'+ sysargv1 + ' encoder, ' + sysargv2
#################################################
if config.getint('settings','gpu'):
    import tensorflow as tf
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)


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
        
        for n in dictionary.loaded[k].AllNames:
            concept_ids.append(c_id)
            # concept_all_ids.append(a_ids)
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
positives_training = prepare_positives(positives_training,nltk.word_tokenize,vocabulary)
positives_dev = prepare_positives(positives_dev,nltk.word_tokenize,vocabulary)
positives_dev_truncated = prepare_positives(positives_dev_truncated,nltk.word_tokenize,vocabulary)


# corpus
corpus_train = sample.NewDataSet('training corpus')
corpus_train.objects = load.load(os.path.normpath(config['corpus']['training_file']),'NCBI')

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
    corpus.tokenize = [nltk.word_tokenize(name) for name in corpus.names]
    corpus.vectorize = np.array([[vocabulary.get(text.lower(),1) for text in mention] for mention in corpus.tokenize])
    corpus.padded = pad_sequences(corpus.vectorize, padding='post', maxlen=int(config['embedding']['length']))
    corpus.tokenize = None
    corpus.vectorize = None


# corpus_dev = sample.NewDataSet('dev corpus')
# corpus_dev.objects = load.load(config['corpus']['development_file'],'NCBI')

# for corpus in [corpus_dev]:
#     corpus.ids = [] # list of all ids (gold standard for each mention)
#     corpus.names = [] # list of all names
#     corpus.all = [] # list of tuples (mention_text,gold,context,(start,end,docid))

#     #sth wrong here that sometimes throw an error
#     #import pdb;pdb.set_trace()
#     for abstract in corpus.objects:
#         for section in abstract.sections: # title and abstract
#             for mention in section.mentions:
#                 nor_ids = [sample._nor_id(one_id) for one_id in mention.id]
#                 corpus.ids.append(nor_ids) # append list of ids, usually len(list)=1
#                 corpus.names.append(mention.text)
#                 corpus.all.append((mention.text,nor_ids,section.text,(mention.start,mention.end,abstract.docid)))
#     #corpus.tokenize = [nltk.word_tokenize(name) for name in corpus.names]
#     #corpus.vectorize = np.array([[vocabulary.get(text.lower(),1) for text in mention] for mention in corpus.tokenize])
#     #corpus.padded = pad_sequences(corpus.vectorize, padding='post', maxlen=int(config['embedding']['length']))
#     #corpus.tokenize = None
#     #corpus.vectorize = None


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

def _predict_shared_encoder_nosim(original_model,entity_encodings,concept_encodings):
    layerss = ['v_sem','hidden_layer','prediction_layer']
    d1 = original_model.get_layer(layerss[1])
    d2 = original_model.get_layer(layerss[2])
    concatenate_list = [entity_encodings,concept_encodings]
    join_layer = Concatenate()(concatenate_list)
    hidden_layer = Dense(d1.units, activation=d1.activation,weights=d1.get_weights())(join_layer)
    prediction_layer = Dense(d2.units, activation=d2.activation,weights=d2.get_weights())(hidden_layer)
    model = Model(inputs=[entity_encodings,concept_encodings], outputs=prediction_layer)
    return model

def _predict_shared_encoder(original_model,entity_encodings,concept_encodings):
    layerss = ['v_sem','hidden_layer','prediction_layer']
    v_sem = original_model.get_layer(layerss[0])
    d1 = original_model.get_layer(layerss[1])
    d2 = original_model.get_layer(layerss[2])
    sem = cnn.semantic_similarity_layer(weights = v_sem.get_weights())([entity_encodings,concept_encodings])
    concatenate_list = [entity_encodings,concept_encodings,sem]
    join_layer = Concatenate()(concatenate_list)
    hidden_layer = Dense(d1.units, activation=d1.activation,weights=d1.get_weights())(join_layer)
    prediction_layer = Dense(d2.units, activation=d2.activation,weights=d2.get_weights())(hidden_layer)
    model = Model(inputs=[entity_encodings,concept_encodings], outputs=prediction_layer)
    return model

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

def _predict_shared_encoder_xDense(original_model,entity_encodings,concept_encodings):
    layerss = ['v_sem','hidden_layer','prediction_layer']
    try:
        v_sem = original_model.get_layer(layerss[0])
        d2 = original_model.get_layer(layerss[2])
        sem = cnn.semantic_similarity_layer(weights = v_sem.get_weights())([entity_encodings,concept_encodings])
        prediction_layer = Dense(d2.units, activation=d2.activation,weights=d2.get_weights())(sem)
    except ValueError: # no decision layer
        v_sem = original_model.get_layer(layerss[2])
        prediction_layer = cnn.semantic_similarity_layer(weights = v_sem.get_weights())([entity_encodings,concept_encodings])
    model = Model(inputs=[entity_encodings,concept_encodings], outputs=prediction_layer)
    return model

def _predict_shared_encoder_dot_xDense(original_model,entity_encodings,concept_encodings):
    layerss = ['v_sem','hidden_layer','prediction_layer']
    try:
        d2 = original_model.get_layer(layerss[2])
        sim = layers.dot([entity_encodings, concept_encodings], axes=-1, normalize=True)
        prediction_layer = Dense(d2.units, activation=d2.activation,weights=d2.get_weights())(sim)
    except AttributeError: # no decision layer
        prediction_layer = layers.dot([entity_encodings, concept_encodings], axes=-1, normalize=True)
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

    if sysargv1 == 'no_sim':
        model = _predict_shared_encoder_nosim(original_model,entity_encodings,concept_encodings)
    elif sysargv2 == 'full':
        if sysargv1 == 'sem_matrix':
            model = _predict_shared_encoder(original_model,entity_encodings,concept_encodings)
        elif sysargv1 == 'cosine_sim':
            model = _predict_shared_encoder_dot(original_model,entity_encodings,concept_encodings)
    elif sysargv2 == 'ablation':
        if sysargv1 == 'sem_matrix':
            model = _predict_shared_encoder_xDense(original_model,entity_encodings,concept_encodings)
        elif sysargv1 == 'cosine_sim':
            model = _predict_shared_encoder_dot_xDense(original_model,entity_encodings,concept_encodings)

    test_y = model.predict(convoluted_input)

    if not result:
        evaluation_parameter = callback.evaluate(val_data.mentions, test_y, val_data.y)
    else:
        evaluation_parameter = evaluate_w_results(val_data.mentions, test_y, val_data.y, concept, result)

    return evaluation_parameter


from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from keras.callbacks import Callback

class EarlyStoppingRankingAccuracyGenerator(Callback):
    ''' Ranking accuracy callback with early stopping.

    '''
    def __init__(self, conf, concept, positives, vocab, entity_model, concept_model, original_model,val_data):
        super().__init__()
        #raise NotImplementedError('The evaluation somehow does not work.')
        import callback
        self.conf = conf
        self.concept = concept
        self.positives = positives
        self.vocab = vocab
        self.entity_model = entity_model
        self.concept_model = concept_model
        self.original_model = original_model
        self.val_data = val_data

        self.best = 0 # best accuracy
        self.wait = 0
        self.stopped_epoch = 0
        self.patience = int(conf['training']['patience'])
        self.model_path = conf['model']['path_model_whole']

        self.save = int(self.conf['settings']['save_prediction'])
        self.now = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.history = self.conf['settings']['history'] + self.now + '.txt'
        callback.write_training_info(self.conf,self.history)

    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracy = []

        self.wait = 0
        with open(self.history,'a',encoding='utf-8') as fh:
        # Pass the file handle in as a lambda function to make it callable
            self.original_model.summary(print_fn=lambda x: fh.write(x + '\n'))
        return

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        evaluation_parameter = predict(self.conf, self.concept, self.positives, self.vocab, self.entity_model, self.concept_model,self.original_model, self.val_data)
        self.accuracy.append(evaluation_parameter)

        with open(self.history,'a',encoding='utf-8') as f:
            f.write('Epoch: {0}, Training loss: {1}, validation accuracy: {2}\n'.format(epoch,logs.get('loss'),evaluation_parameter))
            if logs.get('val_loss'):
                f.write('Epoch: {0}, Validation loss: {1}\n'.format(epoch,logs.get('val_loss')))
                
        if evaluation_parameter > self.best:
            logging.info('Intermediate model saved.')
            self.best = evaluation_parameter
            self.original_model.save(self.model_path)
            self.wait = 0
            # something here to print trec_eval doc
        else:
            self.wait += 1
            if self.wait > int(self.conf['training']['patience']):
                self.stopped_epoch = epoch
                self.original_model.stop_training = True
        # if self.save and self.original_model.stop_training:
        #     logger.info('Saving predictions to {0}'.format(self.conf['model']['path_saved_predictions']))
        #     model_tools.save_predictions(self.conf['model']['path_saved_predictions'],test_y) #(filename,predictions)
        logger.info('Testing: epoch: {0}, self.original_model.stop_training: {1}'.format(epoch,self.original_model.stop_training))
        return

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            logger.info('Epoch %05d: early stopping', self.stopped_epoch + 1)
        try:
            self.original_model.load_weights(self.model_path)
            logger.info('Best model reloaded.')
        except OSError:
            pass
        predict(self.conf, self.concept, self.positives, self.vocab, self.entity_model, self.concept_model,self.original_model, self.val_data, result=self.history)
        if self.conf.getint('model','save'):
            callback.save_model(self.original_model, self.conf['model']['path'],self.now)
        return

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        return


from model_tools import load_model, save_model
import cnn, model_tools

if sysargv1 == 'no_sim':
    model, entity_model, concept_model = cnn.build_model_shared_encoder_nosim(config,vocabulary,pretrained)
elif sysargv2 == 'full':
    if sysargv1 == 'sem_matrix':
        model, entity_model, concept_model = cnn.build_model_generator(config,vocabulary,pretrained)
    elif sysargv1 == 'cosine_sim':
        model, entity_model, concept_model = cnn.build_model_shared_encoder_dot(config,tr_data,vocabulary,pretrained)
    else:
        raise ValueError
elif sysargv2 == 'ablation':
    if sysargv1 == 'sem_matrix':
        model, entity_model, concept_model = cnn.build_model_shared_encoder_xDense(config,tr_data,vocabulary,pretrained,decision_layer=True)
    elif sysargv1 == 'cosine_sim':
        model, entity_model, concept_model = cnn.build_model_shared_encoder_dot_xDense(config,tr_data,vocabulary,pretrained,decision_layer=False)
    else:
        raise ValueError
else:
    raise ValueError

dev_eval_function = EarlyStoppingRankingAccuracyGenerator(config, concept, positives_dev_truncated, vocabulary, entity_model, concept_model, model, real_val_data)

train_examples = examples(config, concept, positives_training, vocabulary)
dev_examples = examples(config, concept, positives_dev, vocabulary)

hist = model.fit_generator(train_examples, steps_per_epoch=len(corpus_train.names), validation_data=dev_examples, validation_steps=len(corpus_dev.names), epochs=config.getint('training','epoch'), callbacks=[dev_eval_function])

#import pdb; pdb.set_trace()