'''
Shared encoder, generator
test embedding by pre-training
give 50, 200, 400 for the embedding used as the first argument
'''

pretrain = True

import logging
import logging.config

import configparser as cp
import sys

import numpy as np
from keras.preprocessing.sequence import pad_sequences
import pickle
import random
import nltk
import time
import os

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
#config['embedding']['emb_file'] = '/home/lenz/disease-normalization/data/embeddings/wvec_50_haodi-li-et-al.bin'
#'/home/lhchan/disease_normalization/data/pubmed2018_w2v_400D/pubmed2018_w2v_400D.bin'
#'/home/lenz/disease-normalization/data/embeddings/wvec_50_haodi-li-et-al.bin'
config['cnn']['filters'] = '50'
config['cnn']['optimizer'] = 'adam'
config['cnn']['lr'] = '0.00005'
config['cnn']['loss'] = 'binary_crossentropy'
config['cnn']['dropout'] = '0.5'
config['embedding']['length'] = '10'
config['embedding']['limit'] = '1000000'
config['note']['note'] = 'pretraining with synonyms p=10, d=200 (continue pretraining from saved model 20190416-232428'
config['model']['save'] = '1'
#################################################

if sys.argv[1]=='50':
    config['embedding']['emb_file'] = os.path.join(directory, '../../../lenz/disease-normalization/data/embeddings/wvec_50_haodi-li-et-al.bin')
elif sys.argv[1]=='200':
    config['embedding']['emb_file'] = os.path.join(directory, '../../disease_normalization/data/pubmed2018_w2v_200D/pubmed2018_w2v_200D.bin')
elif sys.argv[1]=='400':
    config['embedding']['emb_file'] = os.path.join(directory, '../../disease_normalization/data/pubmed2018_w2v_400D/pubmed2018_w2v_400D.bin')

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
dictionary.loaded = load.load(config['terminology']['dict_file'],'MEDIC')


def concept_obj(conf,dictionary,order=None):
    # concept_ids = [] # list of all concept ids
    # concept_all_ids = [] # list of (lists of all concept ids with alt IDs)
    concept_names = [] # list of all names, same length as concept_ids
    # concept_map = {} # names as keys, ids as concepts


    if order:
        use = order
        logger.info('Re-initializng concept object.')
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

    # save the stuff to object
    concept = sample.NewDataSet('concepts')
    #concept.ids = concept_ids
    #concept.all_ids = concept_all_ids
    concept.names = concept_names
    #concept.map = concept_map
    concept.tokenize = [nltk.word_tokenize(name) for name in concept.names] # list of list of tokenized names
    concept.vectorize = np.array([[vocabulary.get(text.lower(),1) for text in concept] for concept in concept.tokenize])
    concept.padded = pad_sequences(concept.vectorize, padding='post', maxlen=int(config['embedding']['length']))

    return concept


def generate_synonym_pairs(dictionary, order=None):
    concept_synonyms = []

    if order:
        use = order
        logger.info('Re-initializing concept object.')
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

# validation set
[real_val_data,concept_order] = pickle.load(open(os.path.join(directory, 'gitig_real_val_data.pickle'),'rb'))
real_val_data.y=np.array(real_val_data.y)
real_val_data.x = None

concept = concept_obj(config,dictionary,order=concept_order)


# corpus
corpus_train = sample.NewDataSet('training corpus')
corpus_train.objects = load.load(os.path.normpath(config['corpus']['training_file']),'NCBI')

corpus_dev = sample.NewDataSet('dev corpus')
corpus_dev.objects = load.load(config['corpus']['development_file'],'NCBI')

for corpus in [corpus_train, corpus_dev]:
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

corpus_dev.tokenize = [nltk.word_tokenize(name) for name in corpus_dev.names]
corpus_dev.vectorize = np.array([[vocabulary.get(text.lower(),1) for text in mention] for mention in corpus_dev.tokenize])
corpus_dev.padded = pad_sequences(corpus_dev.vectorize, padding='post', maxlen=int(config['embedding']['length']))
corpus_dev.tokenize = None
corpus_dev.vectorize = None


if pretrain:
    synonym_pairs = generate_synonym_pairs(dictionary,order=concept_order)
    questions = [question for question, answer in synonym_pairs]
    answers = [answer for question, answer in synonym_pairs]
    # FIXME: there may be positives as well
    # negatives = random.choices(concept.names,k=len(questions)) # this only works for python 3.6 +
    negatives = [random.choice(concept.names) for i in range(len(questions))]


    collection = []
    for question, positive, negative in zip(questions,answers,negatives):
        collection.extend([(question,positive,1),(question,negative,0)])
    random.shuffle(collection)
    tr_data = sample.Data()


    for sat, data in zip([collection],[tr_data]):
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
    from callback import EarlyStoppingRankingAccuracySpedUpGiveModel, Timed
    import cnn, model_tools

    # pretraining with shared encoders
    evaluation_function = EarlyStoppingRankingAccuracySpedUpGiveModel(config,real_val_data,concept.padded,corpus_dev.padded,pretrained,cnn.forward_pass_speedup_shared_encoder)
    timing_epoch = Timed()

    model_shared_encoder, entity_model, concept_model = cnn.build_model_shared_encoder(config,tr_data,vocabulary,pretrained)
    model_shared_encoder.load_weights('../gitig/model_whole_20190416-232428.h5')

    #del vocabulary
    hist_shared = model_shared_encoder.fit(tr_data.x, tr_data.y, epochs=77, batch_size=100,callbacks=[evaluation_function,timing_epoch])
    import pdb; pdb.set_trace()


# continue training with real data
from sample import prepare_positives
positives_training, positives_dev, positives_dev_truncated = pickle.load(open(os.path.join(directory, 'gitig_positive_indices.pickle'),'rb'))
# positives = pickle.load(open('gitig_positive_indices_all.pickle','rb'))
positives_training = prepare_positives(positives_training,nltk.word_tokenize,vocabulary)
positives_dev_truncated = prepare_positives(positives_dev_truncated,nltk.word_tokenize,vocabulary)


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


    corpus.ids = mention_ids
    corpus.names = mention_names
    corpus.all = mention_all
    corpus.tokenize = mention_tokenize
    corpus.vectorize = mention_vectorize

# padding
for corpus in [corpus_train]:
    logger.info('Padding {0}'.format(corpus.info))
    logger.info('Old shape: {0}'.format(corpus.vectorize.shape))
    corpus.padded = pad_sequences(corpus.vectorize, padding='post', maxlen=int(config['embedding']['length']))
    #format of corpus.padded: numpy, mentions, padded


# import sp_training
# def sampling(conf,positives,concept,corpus_train_padded):
#     logger.info('Sampling training examples...')
#     sampled = [sp_training.sample_for_individual_mention(pos,len(concept.names),config.getint('sample','neg_count')) for pos,men in positives]
#     name_order = [men for pos,men in positives]
#     tr_data = sample.Data()
#     tr_data.mentions = sample.sample_format_mentions(sampled,name_order)
#     tr_data.x = sample.sample_format_x(sampled,corpus_train.padded,concept.padded,tr_data.mentions)
#     tr_data.y = sample.sample_format_y(sampled)
#     assert len(tr_data.x[0]) == len(tr_data.y)
#     return tr_data
# tr_data = sampling(config,positives,concept,corpus_train.padded)


from sample import examples
train_examples = examples(config, concept, positives_training, vocabulary)
dev_examples = examples(config, concept, positives_dev_truncated, vocabulary)


#from callback import EarlyStoppingRankingAccuracySpedUpSharedEncoder
    
#config['note']['note'] = 'continue from pretraining on synpair (20190401-120420.txt)'
import cnn, model_tools

# evaluation_function_truncated_dev = EarlyStoppingRankingAccuracySpedUpSharedEncoder(config,real_val_data,concept.padded,corpus_dev.padded,pretrained)
# from datetime import datetime

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

def build_model_generator(conf,vocabulary,pretrained):
    inp_mentions = Input(shape=(conf.getint('embedding','length'),),dtype='int32', name='inp_mentions')
    inp_candidates = Input(shape=(conf.getint('embedding','length'),),dtype='int32', name='inp_candidates')

    embedding_layer = Embedding(len(vocabulary), pretrained.shape[1], mask_zero=False, trainable=False, weights=[pretrained], name='embedding_layer')
    drop = layers.Dropout(conf.getfloat('cnn','dropout'),name='drop')
    encoded_mentions = drop(embedding_layer(inp_mentions))
    encoded_candidates = drop(embedding_layer(inp_candidates))

    SharedConv = Conv1D(filters=conf.getint('cnn','filters'),kernel_size=conf.getint('cnn','kernel_size'),activation='relu')
    conv_mentions = SharedConv(encoded_mentions)
    conv_candidates = SharedConv(encoded_candidates)

    pooled_mentions = GlobalMaxPooling1D()(conv_mentions)
    pooled_candidates = GlobalMaxPooling1D()(conv_candidates)

    entity_model = Model(inputs=inp_mentions, outputs=pooled_mentions)
    concept_model = Model(inputs=inp_candidates, outputs=pooled_candidates)

    v_sem = semantic_similarity_layer(name='v_sem')([pooled_mentions,pooled_candidates])

    # list of layers for concatenation
    concatenate_list = [pooled_mentions,pooled_candidates,v_sem]

    join_layer = Concatenate()(concatenate_list)
    hidden_layer = Dense(64, activation='relu',name='hidden_layer')(join_layer)
    prediction_layer = Dense(1,activation='sigmoid',name='prediction_layer')(hidden_layer)  

    # list of input layers
    input_list = [inp_mentions,inp_candidates]

    model = Model(inputs=input_list, outputs=prediction_layer)
    model.compile(optimizer=cnn.return_optimizer(conf), loss=cnn.return_loss(conf))

    return model, entity_model, concept_model


def predict(config, concept, positives, vocab, entity_model, concept_model, original_model,val_data,result=None):
    entity_examples = examples(config, concept, positives, vocab, neg_count=0)

    #c_token_indices = [[vocab.get(t.lower(), 1) for t in nltk.word_tokenize(neg)] for neg in concept.names]
    concept_examples = pad_sequences(concept.vectorize, maxlen=config.getint('embedding','length'))
   
    entity_encodings = entity_model.predict_generator(entity_examples, steps=len(positives))    
    concept_encodings = concept_model.predict(concept_examples)

    ###################
    from sample import sped_up_format_x
    convoluted_input = sped_up_format_x(entity_encodings,concept_encodings)
    
    layerss = ['v_sem','hidden_layer','prediction_layer']
    v_sem = original_model.get_layer(layerss[0])
    d1 = original_model.get_layer(layerss[1])
    d2 = original_model.get_layer(layerss[2])

    entity_encodings = Input(shape=(convoluted_input[0].shape[1],),dtype='float32', name='entity_encodings')
    concept_encodings = Input(shape=(convoluted_input[1].shape[1],),dtype='float32', name='concept_encodings')
    sem = cnn.semantic_similarity_layer(weights = v_sem.get_weights())([entity_encodings,concept_encodings])
    concatenate_list = [entity_encodings,concept_encodings,sem]
    join_layer = Concatenate()(concatenate_list)
    hidden_layer = Dense(d1.units, activation=d1.activation,weights=d1.get_weights())(join_layer)
    prediction_layer = Dense(d2.units, activation=d2.activation,weights=d2.get_weights())(hidden_layer)

    model = Model(inputs=[entity_encodings,concept_encodings], outputs=prediction_layer)
    test_y = model.predict(convoluted_input)
    if not result:
        import callback
        evaluation_parameter = callback.evaluate(val_data.mentions, test_y, val_data.y)
    else:
        evaluation_parameter = evaluate_w_results(val_data.mentions, test_y, val_data.y, concept, result)
    ###################
    # sims = cosine_similarity(entity_encodings, concept_encodings)
    
    # best_hits = np.argmax(sims, axis=-1)
    # predictions = [concept.ids[i] for i in best_hits]
    
    # return predictions
    return evaluation_parameter

from datetime import datetime
from keras.callbacks import Callback
class EarlyStoppingRankingAccuracyGenerator(Callback):
    ''' Ranking accuracy callback with early stopping.

    '''
    def __init__(self, conf, concept, positives, vocab, entity_model, concept_model, original_model,val_data):
        super().__init__()
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

        evaluation_parameter = predict(self.conf, self.concept, self.positives, self.vocab, self.entity_model, self.concept_model,self.model, self.val_data)
        self.accuracy.append(evaluation_parameter)

        with open(self.history,'a',encoding='utf-8') as f:
            f.write('Epoch: {0}, Training loss: {1}, validation accuracy: {2}\n'.format(epoch,logs.get('loss'),evaluation_parameter))

        if evaluation_parameter > self.best:
            logging.info('Intermediate model saved.')
            self.best = evaluation_parameter
            self.model.save(self.model_path)
            self.wait = 0
            # something here to print trec_eval doc
        else:
            self.wait += 1
            if self.wait > int(self.conf['training']['patience']):
                self.stopped_epoch = epoch
                self.model.stop_training = True
        if self.save and self.model.stop_training:
            logger.info('Saving predictions to {0}'.format(self.conf['model']['path_saved_predictions']))
            model_tools.save_predictions(self.conf['model']['path_saved_predictions'],test_y) #(filename,predictions)
        logger.info('Testing: epoch: {0}, self.model.stop_training: {1}'.format(epoch,self.model.stop_training))
        return

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            logging.info('Epoch %05d: early stopping', self.stopped_epoch + 1)
        try:
            from cnn import semantic_similarity_layer, ranking_loss
            from keras.models import load_model
            self.model = load_model(self.model_path,custom_objects={'semantic_similarity_layer': semantic_similarity_layer, 'ranking_loss':ranking_loss})
        except OSError:
            pass
        predict(self.conf, self.concept, self.positives, self.vocab, self.entity_model, self.concept_model,self.model, self.val_data, result=self.history)
        if self.conf.getint('model','save'):
            callback.save_model(self.model, self.conf['model']['path'],self.now)
        return

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        return


dummy = None

model, entity_model, concept_model = cnn.build_model_shared_encoder(config,dummy,vocabulary,pretrained)
model.load_weights('models/20190407-105603.h5')
# model_shared_encoder = model_tools.load_model('models/20190407-105603.json','models/20190407-105603.h5',{'semantic_similarity_layer': semantic_similarity_layer})

evaluation_function = EarlyStoppingRankingAccuracyGenerator(config, concept, positives_dev_truncated, vocabulary, entity_model, concept_model, model, real_val_data)

# from keras import optimizers
# adam = optimizers.Adam(lr=0.00001, epsilon=None, decay=0.0)
# model_shared_encoder.compile(optimizer=adam,loss='binary_crossentropy')


for ep in range(100):
    print('Epoch: {0}'.format(ep+1))
    for i in range(config.getint('training','epoch')):
        model.fit_generator(train_examples, steps_per_epoch=len(corpus_train.names), validation_data=dev_examples, validation_steps=len(corpus_dev.names), epochs=1, callbacks=[evaluation_function])

import pdb; pdb.set_trace()

