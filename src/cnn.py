#!/usr/bin/env python3
# coding: utf8

from keras.models import Model
from keras.layers import Input, Embedding, Concatenate, Dense, Conv1D, GlobalMaxPooling1D, Flatten
from keras import layers

import logging
logger = logging.getLogger(__name__)

def return_optimizer(conf):
    from keras import optimizers
    if conf.getfloat('cnn','lr'):
        lr = conf.getfloat('cnn','lr')
        adam = optimizers.Adam(lr=lr, epsilon=None, decay=0.0) # default lr=0.001
        adagrad = optimizers.Adagrad(lr=lr, epsilon=None, decay=0.0)
        adadelta = optimizers.Adadelta(lr=lr, rho=0.95, epsilon=None, decay=0.0)    
    else:
        adam = optimizers.Adam(lr=0.001, epsilon=None, decay=0.0) # default lr=0.001
        adagrad = optimizers.Adagrad(lr=0.001, epsilon=None, decay=0.0)
        adadelta = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    opts = {'adam':adam,'adagrad':adagrad,'adadelta':adadelta}
    return opts[conf['cnn']['optimizer']]

def return_loss(conf):
    opts = {'binary_crossentropy':'binary_crossentropy','ranking_loss':ranking_loss}
    return opts[conf['cnn']['loss']]

def print_input(training_data):
    for i in range(len(training_data.x)):
        logger.info('training_data.x[{0}]\tshape:{1}\tdtype:{2}'.format(i,training_data.x[i].shape,training_data.x[i].dtype))

from keras import backend as K
from keras.engine.topology import Layer

class semantic_similarity_layer(Layer):
    '''
    Join layer with a trainable similarity matrix.
    v_sem = sim(v_m, v_c) = v_m^T M v_c
    input shape = [(m_0,m_1),(c_0,c_1)]
    '''
    def __init__(self, **kwargs):
        self.M = None
        super().__init__(**kwargs)

    def build(self, input_shape):
        try:
            shape_m, shape_c = input_shape
            assert len(shape_m)==len(shape_c) & len(shape_m) == 2
        except ValueError:
            raise ValueError('input_shape must be a 2-element list, each containing a two-element tuple.')

        import keras.initializers
        #ini = keras.initializers.Identity(gain=1.0)
        ini = 'uniform'
        self.M = self.add_weight(name='M',
                                 shape=(shape_m[1], shape_c[1]),
                                 initializer=ini,
                                 trainable=True)
        import numpy as np
        #assert self.M.dtype is np.dtype('float32')
        super().build(input_shape)

    def call(self, x):
        '''
        x: list consisting of outputs from the two pooling layers
        https://github.com/wglassly/cnnormaliztion/blob/master/src/nn_layers.py#L822
        https://github.com/lfurrer/disease-normalization/blob/master/tzlink/rank/cnn.py
        '''
        try:
            m, c = x
        except ValueError:
            raise ValueError('Input must be a list of outputs from the previous two layers.')
        return K.batch_dot(m, K.dot(c, K.transpose(self.M)), axes=1)

    def compute_output_shape(self, input_shape):
        '''
        output is the main diagonal of x dot y.T -> shape is (x[0],1)
        '''
        assert isinstance(input_shape, list)
        shape_m, shape_c = input_shape
        return (shape_m[0], 1)

'''
    def get_config(self):
        config = {'batch_input_shape': self.batch_input_shape,
                  'dtype': self.dtype,
                  'sparse': self.sparse,
                  'name': self.name}
        return config
'''

def ranking_loss(y_true, y_pred):
    # y_true is meaningless for us.
    from keras import backend as K
    pos = K.tile(y_pred[:1], (K.shape(y_pred)[0]-1, 1)) # +1 is just a bias 
    margin = K.mean(pos-y_pred[1:])
    return -margin # We want to maximize margin

def build_model(conf,training_data,vocabulary,pretrained):
    inp_mentions = Input(shape=(training_data.x[0].shape[1],),dtype='int32', name='inp_mentions')
    inp_candidates = Input(shape=(training_data.x[1].shape[1],),dtype='int32', name='inp_candidates')
    if int(conf['candidate']['use']):
        inp_scores = Input(shape=(training_data.x[2].shape[1],),dtype='float64', name='inp_scores')
    if conf.getint('embedding','elmo'):
        inp_mentions_elmo = Input(shape=(training_data.x[-2].shape[1],),dtype='float32', name='inp_mentions_elmo')
        inp_candidates_elmo = Input(shape=(training_data.x[-1].shape[1],),dtype='float32', name='inp_candidates_elmo')

    embedding_layer = Embedding(len(vocabulary), pretrained.shape[1], mask_zero=False, trainable=False, weights=[pretrained], name='embedding_layer')
    drop = layers.Dropout(conf.getfloat('cnn','dropout'),name='drop')
    encoded_mentions = drop(embedding_layer(inp_mentions))
    encoded_candidates = drop(embedding_layer(inp_candidates))

    conv_mentions = Conv1D(filters=conf.getint('cnn','filters'),kernel_size=conf.getint('cnn','kernel_size'),activation='relu',name='conv_mentions')(encoded_mentions) #input_shape=(2000,16,50)
    conv_candidates = Conv1D(filters=conf.getint('cnn','filters'),kernel_size=conf.getint('cnn','kernel_size'),activation='relu',name='conv_candidates')(encoded_candidates) #input_shape=(2000,16,50)
    pooled_mentions = GlobalMaxPooling1D()(conv_mentions)
    pooled_candidates = GlobalMaxPooling1D()(conv_candidates)
    if conf.getint('embedding','elmo'):
        conv_mentions_elmo = Conv1D(filters=conf.getint('cnn','filters'),kernel_size=conf.getint('cnn','kernel_size'),activation='relu')(inp_mentions_elmo)
        conv_candidates_elmo = Conv1D(filters=conf.getint('cnn','filters'),kernel_size=conf.getint('cnn','kernel_size'),activation='relu')(inp_candidates_elmo)
        pooled_mentions_elmo = GlobalMaxPooling1D()(conv_mentions_elmo)
        pooled_candidates_elmo = GlobalMaxPooling1D()(conv_candidates_elmo)
        v_sem_elmo = semantic_similarity_layer()([pooled_mentions_elmo,pooled_candidates_elmo])

    v_sem = semantic_similarity_layer(name='v_sem')([pooled_mentions,pooled_candidates])

    # list of layers for concatenation
    concatenate_list = [pooled_mentions,pooled_candidates,v_sem]
    if int(conf['candidate']['use']):
        concatenate_list.append(inp_scores)
    if conf.getint('embedding','elmo'):
        concatenate_list.extend([pooled_mentions_elmo,pooled_candidates_elmo,v_sem_elmo])

    join_layer = Concatenate()(concatenate_list)
    hidden_layer = Dense(64, activation='relu',name='hidden_layer')(join_layer)
    if int(conf['settings']['imp_tr']):
        from keras.layers import Activation
        prediction_layer = Activation('sigmoid')(hidden_layer)
    else:
        prediction_layer = Dense(1,activation='sigmoid',name='prediction_layer')(hidden_layer)  

    # list of input layers
    input_list = [inp_mentions,inp_candidates]
    if int(conf['candidate']['use']):
        input_list.append(inp_scores)
    if conf.getint('embedding','elmo'):
        input_list.extend([inp_mentions_elmo,inp_candidates_elmo])

    model = Model(inputs=input_list, outputs=prediction_layer)
    model.compile(optimizer=return_optimizer(conf), loss=return_loss(conf))

    return model

'''
>>> model.summary()
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
inp_mentions (InputLayer)       (None, 20)           0                                            
__________________________________________________________________________________________________
inp_candidates (InputLayer)     (None, 20)           0                                            
__________________________________________________________________________________________________
embedding_1 (Embedding)         (None, 20, 400)      400000800   inp_mentions[0][0]               
                                                                 inp_candidates[0][0]             
__________________________________________________________________________________________________
conv1d_1 (Conv1D)               (None, 18, 50)       60050       embedding_1[0][0]                
__________________________________________________________________________________________________
conv1d_2 (Conv1D)               (None, 18, 50)       60050       embedding_1[1][0]                
__________________________________________________________________________________________________
global_max_pooling1d_1 (GlobalM (None, 50)           0           conv1d_1[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_2 (GlobalM (None, 50)           0           conv1d_2[0][0]                   
__________________________________________________________________________________________________
semantic_similarity_layer_1 (se (None, 1)            2500        global_max_pooling1d_1[0][0]     
                                                                 global_max_pooling1d_2[0][0]     
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 101)          0           global_max_pooling1d_1[0][0]     
                                                                 global_max_pooling1d_2[0][0]     
                                                                 semantic_similarity_layer_1[0][0]
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 64)           6528        concatenate_1[0][0]              
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 1)            65          dense_1[0][0]                    
==================================================================================================
Total params: 400,129,993
Trainable params: 129,193
Non-trainable params: 400,000,800
__________________________________________________________________________________________________
'''


# Model to speed up forward pass, used in callback for evaluation

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
    drop = original_model.get_layer(layerss[2])
    conv = original_model.get_layer(layerss[3])

    new_input_terms = Input(shape=(terms.input_shape[1],),dtype='int32', name='new_input_terms')
    new_emb = Embedding(emb.input_dim, emb.output_dim, mask_zero=False, trainable=False, weights=[pretrained])
    new_drop = layers.Dropout(drop.get_config()['rate'])
    encoded = new_drop(new_emb(new_input_terms))
    new_conv = Conv1D(filters=conv.filters,kernel_size=conv.kernel_size[0],activation=conv.activation,weights=conv.get_weights())(encoded)
    gl_max_p = GlobalMaxPooling1D()(new_conv)

    model_part = Model(inputs=new_input_terms, outputs=gl_max_p)
    return model_part

def _forward_pass_speedup_sem(original_model,convoluted_x):
    layers = ['v_sem','hidden_layer','prediction_layer']
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

def forward_pass_speedup(model,corpus_padded,concept_padded,pretrained):
    '''
    Model to speed up forward pass, used in callback for evaluation
    '''
    model_mention = _forward_pass_speedup_conv(model,['inp_mentions','embedding_layer','drop','conv_mentions','global_max_pooling1d_1'],pretrained)
    mentions = model_mention.predict(corpus_padded) # (787, 50)
    model_candidate = _forward_pass_speedup_conv(model,['inp_candidates','embedding_layer','drop','conv_candidates','global_max_pooling1d_2'],pretrained)
    candidates = model_candidate.predict(concept_padded) # (67782,50)
    logger.info('Formatting pooled mentions and candidates...')
    # from sample import no_cangen_format_x
    from sample import sped_up_format_x
    convoluted_input = sped_up_format_x(mentions,candidates)
    model_sem = _forward_pass_speedup_sem(model,convoluted_input)
    return convoluted_input, model_sem



def build_model_maxpool_ablation(conf,training_data,vocabulary,pretrained):
    '''
    Check the effect of taking out the maxpooled mentions and candidates,
    i.e. how the semantic similarity layer along performs.
    '''
    inp_mentions = Input(shape=(training_data.x[0].shape[1],),dtype='int32', name='inp_mentions')
    inp_candidates = Input(shape=(training_data.x[1].shape[1],),dtype='int32', name='inp_candidates')

    embedding_layer = Embedding(len(vocabulary), pretrained.shape[1], mask_zero=False, trainable=False, weights=[pretrained], name='embedding_layer')
    encoded_mentions = embedding_layer(inp_mentions)
    encoded_candidates = embedding_layer(inp_candidates)

    conv_mentions = Conv1D(filters=conf.getint('cnn','filters'),kernel_size=conf.getint('cnn','kernel_size'),activation='relu',name='conv_mentions')(encoded_mentions) #input_shape=(2000,16,50)
    conv_candidates = Conv1D(filters=conf.getint('cnn','filters'),kernel_size=conf.getint('cnn','kernel_size'),activation='relu',name='conv_candidates')(encoded_candidates) #input_shape=(2000,16,50)
    pooled_mentions = GlobalMaxPooling1D()(conv_mentions)
    pooled_candidates = GlobalMaxPooling1D()(conv_candidates)

    v_sem = semantic_similarity_layer(name='v_sem')([pooled_mentions,pooled_candidates])

    prediction_layer = Dense(1,activation='sigmoid',name='prediction_layer')(v_sem)  

    # list of input layers
    input_list = [inp_mentions,inp_candidates]

    model = Model(inputs=input_list, outputs=prediction_layer)
    model.compile(optimizer=return_optimizer(conf), loss=return_loss(conf))

    return model

def build_model_custom_loss(conf,training_data,vocabulary,pretrained,track_obj):
    inp_mentions = Input(shape=(training_data.x[0].shape[1],),dtype='int32', name='inp_mentions')
    inp_candidates = Input(shape=(training_data.x[1].shape[1],),dtype='int32', name='inp_candidates')
    if int(conf['candidate']['use']):
        inp_scores = Input(shape=(training_data.x[2].shape[1],),dtype='float64', name='inp_scores')
    if conf.getint('embedding','elmo'):
        inp_mentions_elmo = Input(shape=(training_data.x[-2].shape[1],),dtype='float32', name='inp_mentions_elmo')
        inp_candidates_elmo = Input(shape=(training_data.x[-1].shape[1],),dtype='float32', name='inp_candidates_elmo')

    embedding_layer = Embedding(len(vocabulary), pretrained.shape[1], mask_zero=False, trainable=False, weights=[pretrained], name='embedding_layer')
    encoded_mentions = embedding_layer(inp_mentions)
    encoded_candidates = embedding_layer(inp_candidates)

    conv_mentions = Conv1D(filters=conf.getint('cnn','filters'),kernel_size=conf.getint('cnn','kernel_size'),activation='relu',name='conv_mentions')(encoded_mentions) #input_shape=(2000,16,50)
    conv_candidates = Conv1D(filters=conf.getint('cnn','filters'),kernel_size=conf.getint('cnn','kernel_size'),activation='relu',name='conv_candidates')(encoded_candidates) #input_shape=(2000,16,50)
    pooled_mentions = GlobalMaxPooling1D()(conv_mentions)
    pooled_candidates = GlobalMaxPooling1D()(conv_candidates)
    if conf.getint('embedding','elmo'):
        conv_mentions_elmo = Conv1D(filters=conf.getint('cnn','filters'),kernel_size=conf.getint('cnn','kernel_size'),activation='relu')(inp_mentions_elmo)
        conv_candidates_elmo = Conv1D(filters=conf.getint('cnn','filters'),kernel_size=conf.getint('cnn','kernel_size'),activation='relu')(inp_candidates_elmo)
        pooled_mentions_elmo = GlobalMaxPooling1D()(conv_mentions_elmo)
        pooled_candidates_elmo = GlobalMaxPooling1D()(conv_candidates_elmo)
        v_sem_elmo = semantic_similarity_layer()([pooled_mentions_elmo,pooled_candidates_elmo])

    v_sem = semantic_similarity_layer(name='v_sem')([pooled_mentions,pooled_candidates])

    # list of layers for concatenation
    concatenate_list = [pooled_mentions,pooled_candidates,v_sem]

    join_layer = Concatenate()(concatenate_list)
    hidden_layer = Dense(64, activation='relu',name='hidden_layer')(join_layer)
    prediction_layer = Dense(1,activation='sigmoid',name='prediction_layer')(hidden_layer)  

    # list of input layers
    input_list = [inp_mentions,inp_candidates]

    model = Model(inputs=input_list, outputs=prediction_layer)
    from nn.loss import AdaptedMaxMarginLoss
    model.compile(optimizer=return_optimizer(conf), loss=AdaptedMaxMarginLoss(track_obj))

    return model


def build_model_shared_encoder(conf,training_data,vocabulary,pretrained):
    inp_mentions = Input(shape=(training_data.x[0].shape[1],),dtype='int32', name='inp_mentions')
    inp_candidates = Input(shape=(training_data.x[1].shape[1],),dtype='int32', name='inp_candidates')

    embedding_layer = Embedding(len(vocabulary), pretrained.shape[1], mask_zero=False, trainable=False, weights=[pretrained], name='embedding_layer')
    drop = layers.Dropout(conf.getfloat('cnn','dropout'),name='drop')
    encoded_mentions = drop(embedding_layer(inp_mentions))
    encoded_candidates = drop(embedding_layer(inp_candidates))

    SharedConv = Conv1D(filters=conf.getint('cnn','filters'),kernel_size=conf.getint('cnn','kernel_size'),activation='relu',name='conv1d')
    conv_mentions = SharedConv(encoded_mentions)
    conv_candidates = SharedConv(encoded_candidates)

    pooled_mentions = GlobalMaxPooling1D()(conv_mentions)
    pooled_candidates = GlobalMaxPooling1D()(conv_candidates)
    if conf.getint('embedding','elmo'):
        conv_mentions_elmo = Conv1D(filters=conf.getint('cnn','filters'),kernel_size=conf.getint('cnn','kernel_size'),activation='relu')(inp_mentions_elmo)
        conv_candidates_elmo = Conv1D(filters=conf.getint('cnn','filters'),kernel_size=conf.getint('cnn','kernel_size'),activation='relu')(inp_candidates_elmo)
        pooled_mentions_elmo = GlobalMaxPooling1D()(conv_mentions_elmo)
        pooled_candidates_elmo = GlobalMaxPooling1D()(conv_candidates_elmo)
        v_sem_elmo = semantic_similarity_layer()([pooled_mentions_elmo,pooled_candidates_elmo])

    v_sem = semantic_similarity_layer(name='v_sem')([pooled_mentions,pooled_candidates])

    # list of layers for concatenation
    concatenate_list = [pooled_mentions,pooled_candidates,v_sem]
    if int(conf['candidate']['use']):
        concatenate_list.append(inp_scores)
    if conf.getint('embedding','elmo'):
        concatenate_list.extend([pooled_mentions_elmo,pooled_candidates_elmo,v_sem_elmo])

    join_layer = Concatenate()(concatenate_list)
    hidden_layer = Dense(64, activation='relu',name='hidden_layer')(join_layer)
    prediction_layer = Dense(1,activation='sigmoid',name='prediction_layer')(hidden_layer)  

    # list of input layers
    input_list = [inp_mentions,inp_candidates]

    model = Model(inputs=input_list, outputs=prediction_layer)
    model.compile(optimizer=return_optimizer(conf), loss=return_loss(conf))

    return model


def forward_pass_speedup_shared_encoder(model,corpus_padded,concept_padded,pretrained):
    '''
    Model to speed up forward pass, used in callback for evaluation
    '''
    model_mention = _forward_pass_speedup_conv(model,['inp_mentions','embedding_layer','drop','conv1d','global_max_pooling1d_1'],pretrained)
    mentions = model_mention.predict(corpus_padded) # (787, 50)
    model_candidate = _forward_pass_speedup_conv(model,['inp_candidates','embedding_layer','drop','conv1d','global_max_pooling1d_2'],pretrained)
    candidates = model_candidate.predict(concept_padded) # (67782,50)
    logger.info('Formatting pooled mentions and candidates...')
    # from sample import no_cangen_format_x
    from sample import sped_up_format_x
    convoluted_input = sped_up_format_x(mentions,candidates)
    model_sem = _forward_pass_speedup_sem(model,convoluted_input)
    return convoluted_input, model_sem

def build_model_shared_encoder_xDense(conf,training_data,vocabulary,pretrained):
    inp_mentions = Input(shape=(training_data.x[0].shape[1],),dtype='int32', name='inp_mentions')
    inp_candidates = Input(shape=(training_data.x[1].shape[1],),dtype='int32', name='inp_candidates')

    embedding_layer = Embedding(len(vocabulary), pretrained.shape[1], mask_zero=False, trainable=False, weights=[pretrained], name='embedding_layer')
    drop = layers.Dropout(conf.getfloat('cnn','dropout'),name='drop')
    encoded_mentions = drop(embedding_layer(inp_mentions))
    encoded_candidates = drop(embedding_layer(inp_candidates))

    SharedConv = Conv1D(filters=conf.getint('cnn','filters'),kernel_size=conf.getint('cnn','kernel_size'),activation='relu')
    conv_mentions = SharedConv(encoded_mentions)
    conv_candidates = SharedConv(encoded_candidates)

    pooled_mentions = GlobalMaxPooling1D()(conv_mentions)
    pooled_candidates = GlobalMaxPooling1D()(conv_candidates)
    if conf.getint('embedding','elmo'):
        conv_mentions_elmo = Conv1D(filters=conf.getint('cnn','filters'),kernel_size=conf.getint('cnn','kernel_size'),activation='relu')(inp_mentions_elmo)
        conv_candidates_elmo = Conv1D(filters=conf.getint('cnn','filters'),kernel_size=conf.getint('cnn','kernel_size'),activation='relu')(inp_candidates_elmo)
        pooled_mentions_elmo = GlobalMaxPooling1D()(conv_mentions_elmo)
        pooled_candidates_elmo = GlobalMaxPooling1D()(conv_candidates_elmo)
        v_sem_elmo = semantic_similarity_layer()([pooled_mentions_elmo,pooled_candidates_elmo])

    v_sem = semantic_similarity_layer(name='v_sem')([pooled_mentions,pooled_candidates])

    prediction_layer = Dense(1,activation='sigmoid',name='prediction_layer')(v_sem)  

    # list of input layers
    input_list = [inp_mentions,inp_candidates]

    model = Model(inputs=input_list, outputs=prediction_layer)
    model.compile(optimizer=return_optimizer(conf), loss=return_loss(conf))

    return model

def _forward_pass_speedup_sem_xDense(original_model,convoluted_x):
    layers = ['v_sem','prediction_layer']
    v_sem = original_model.get_layer(layers[0])
    d1 = original_model.get_layer(layers[1])

    pooled_mentions = Input(shape=(convoluted_x[0].shape[1],),dtype='float32', name='pooled_mentions')
    pooled_candidates = Input(shape=(convoluted_x[1].shape[1],),dtype='float32', name='pooled_candidates')
    sem = semantic_similarity_layer(weights = v_sem.get_weights())([pooled_mentions,pooled_candidates])
    prediction_layer = Dense(d1.units, activation=d1.activation,weights=d1.get_weights())(sem)
    
    input_list = [pooled_mentions, pooled_candidates]
    model_part = Model(inputs=input_list, outputs=prediction_layer)
    return model_part

def forward_pass_speedup_shared_encoder_xDense(model,corpus_padded,concept_padded,pretrained):
    '''
    Model to speed up forward pass, used in callback for evaluation
    '''
    model_mention = _forward_pass_speedup_conv(model,['inp_mentions','embedding_layer','drop','conv1d','global_max_pooling1d_1'],pretrained)
    mentions = model_mention.predict(corpus_padded) # (787, 50)
    model_candidate = _forward_pass_speedup_conv(model,['inp_candidates','embedding_layer','drop','conv1d','global_max_pooling1d_2'],pretrained)
    candidates = model_candidate.predict(concept_padded) # (67782,50)
    logger.info('Formatting pooled mentions and candidates...')
    # from sample import no_cangen_format_x
    from sample import sped_up_format_x
    convoluted_input = sped_up_format_x(mentions,candidates)
    model_sem = _forward_pass_speedup_sem_xDense(model,convoluted_input)
    return convoluted_input, model_sem


def build_model_shared_encoder_dot(conf,training_data,vocabulary,pretrained):
    inp_mentions = Input(shape=(training_data.x[0].shape[1],),dtype='int32', name='inp_mentions')
    inp_candidates = Input(shape=(training_data.x[1].shape[1],),dtype='int32', name='inp_candidates')

    embedding_layer = Embedding(len(vocabulary), pretrained.shape[1], mask_zero=False, trainable=False, weights=[pretrained], name='embedding_layer')
    drop = layers.Dropout(conf.getfloat('cnn','dropout'),name='drop')
    encoded_mentions = drop(embedding_layer(inp_mentions))
    encoded_candidates = drop(embedding_layer(inp_candidates))

    SharedConv = Conv1D(filters=conf.getint('cnn','filters'),kernel_size=conf.getint('cnn','kernel_size'),activation='relu',name='conv1d')
    conv_mentions = SharedConv(encoded_mentions)
    conv_candidates = SharedConv(encoded_candidates)

    pooled_mentions = GlobalMaxPooling1D()(conv_mentions)
    pooled_candidates = GlobalMaxPooling1D()(conv_candidates)

    #v_sem = semantic_similarity_layer()([pooled_mentions,pooled_candidates])
    cos_sim = layers.dot([pooled_mentions, pooled_candidates], axes=-1, normalize=True, name='cos_sim')

    # list of layers for concatenation
    concatenate_list = [pooled_mentions,pooled_candidates,cos_sim]
    if int(conf['candidate']['use']):
        concatenate_list.append(inp_scores)
    if conf.getint('embedding','elmo'):
        concatenate_list.extend([pooled_mentions_elmo,pooled_candidates_elmo,v_sem_elmo])

    join_layer = Concatenate()(concatenate_list)
    hidden_layer = Dense(64, activation='relu',name='hidden_layer')(join_layer)
    prediction_layer = Dense(1,activation='sigmoid',name='prediction_layer')(hidden_layer)  

    # list of input layers
    input_list = [inp_mentions,inp_candidates]

    model = Model(inputs=input_list, outputs=prediction_layer)
    model.compile(optimizer=return_optimizer(conf), loss=return_loss(conf))

    return model

def build_model_shared_encoder_dot_xDense(conf,training_data,vocabulary,pretrained):
    inp_mentions = Input(shape=(training_data.x[0].shape[1],),dtype='int32', name='inp_mentions')
    inp_candidates = Input(shape=(training_data.x[1].shape[1],),dtype='int32', name='inp_candidates')

    embedding_layer = Embedding(len(vocabulary), pretrained.shape[1], mask_zero=False, trainable=False, weights=[pretrained], name='embedding_layer')
    drop = layers.Dropout(conf.getfloat('cnn','dropout'),name='drop')
    encoded_mentions = drop(embedding_layer(inp_mentions))
    encoded_candidates = drop(embedding_layer(inp_candidates))

    SharedConv = Conv1D(filters=conf.getint('cnn','filters'),kernel_size=conf.getint('cnn','kernel_size'),activation='relu',name='conv1d')
    conv_mentions = SharedConv(encoded_mentions)
    conv_candidates = SharedConv(encoded_candidates)

    pooled_mentions = GlobalMaxPooling1D()(conv_mentions)
    pooled_candidates = GlobalMaxPooling1D()(conv_candidates)

    #v_sem = semantic_similarity_layer()([pooled_mentions,pooled_candidates])
    prediction_layer = layers.dot([pooled_mentions, pooled_candidates], axes=-1, normalize=True, name='cos_sim')

    #prediction_layer = Dense(1,activation='sigmoid',name='prediction_layer')(cos_sim)  

    # list of input layers
    input_list = [inp_mentions,inp_candidates]

    model = Model(inputs=input_list, outputs=prediction_layer)
    model.compile(optimizer=return_optimizer(conf), loss=return_loss(conf))

    return model


def build_model_dot(conf,training_data,vocabulary,pretrained):
    inp_mentions = Input(shape=(training_data.x[0].shape[1],),dtype='int32', name='inp_mentions')
    inp_candidates = Input(shape=(training_data.x[1].shape[1],),dtype='int32', name='inp_candidates')

    embedding_layer = Embedding(len(vocabulary), pretrained.shape[1], mask_zero=False, trainable=False, weights=[pretrained], name='embedding_layer')
    drop = layers.Dropout(conf.getfloat('cnn','dropout'),name='drop')
    encoded_mentions = drop(embedding_layer(inp_mentions))
    encoded_candidates = drop(embedding_layer(inp_candidates))

    SharedConv = Conv1D(filters=conf.getint('cnn','filters'),kernel_size=conf.getint('cnn','kernel_size'),activation='relu',name='conv1d')
    conv_mentions = SharedConv(encoded_mentions)
    conv_candidates = SharedConv(encoded_candidates)

    pooled_mentions = GlobalMaxPooling1D()(conv_mentions)
    pooled_candidates = GlobalMaxPooling1D()(conv_candidates)

    #v_sem = semantic_similarity_layer()([pooled_mentions,pooled_candidates])
    cos_sim = layers.dot([pooled_mentions, pooled_candidates], axes=-1, normalize=True, name='cos_sim')

    # list of layers for concatenation
    concatenate_list = [pooled_mentions,pooled_candidates,cos_sim]
    if int(conf['candidate']['use']):
        concatenate_list.append(inp_scores)
    if conf.getint('embedding','elmo'):
        concatenate_list.extend([pooled_mentions_elmo,pooled_candidates_elmo,v_sem_elmo])

    join_layer = Concatenate()(concatenate_list)
    hidden_layer = Dense(64, activation='relu',name='hidden_layer')(join_layer)
    prediction_layer = Dense(1,activation='sigmoid',name='prediction_layer')(hidden_layer)  

    # list of input layers
    input_list = [inp_mentions,inp_candidates]

    model = Model(inputs=input_list, outputs=prediction_layer)
    model.compile(optimizer=return_optimizer(conf), loss=return_loss(conf))

    return model

def _forward_pass_speedup_dot(original_model,convoluted_x):
    layerss = ['hidden_layer','prediction_layer']
    d1 = original_model.get_layer(layerss[0])
    d2 = original_model.get_layer(layerss[1])

    pooled_mentions = Input(shape=(convoluted_x[0].shape[1],),dtype='float32', name='pooled_mentions')
    pooled_candidates = Input(shape=(convoluted_x[1].shape[1],),dtype='float32', name='pooled_candidates')
    cos_sim = layers.dot([pooled_mentions, pooled_candidates], axes=-1, normalize=True, name='cos_sim')
    concatenate_list = [pooled_mentions,pooled_candidates,cos_sim]
    join_layer = Concatenate()(concatenate_list)
    hidden_layer = Dense(d1.units, activation=d1.activation,weights=d1.get_weights())(join_layer)
    prediction_layer = Dense(d2.units, activation=d2.activation,weights=d2.get_weights())(hidden_layer)
    
    input_list = [pooled_mentions, pooled_candidates]
    model_part = Model(inputs=input_list, outputs=prediction_layer)
    return model_part


def forward_pass_speedup_shared_encoder_dot(model,corpus_padded,concept_padded,pretrained):
    '''
    Model to speed up forward pass, used in callback for evaluation
    '''
    model_mention = _forward_pass_speedup_conv(model,['inp_mentions','embedding_layer','drop','conv1d','global_max_pooling1d_1'],pretrained)
    mentions = model_mention.predict(corpus_padded) # (787, 50)
    model_candidate = _forward_pass_speedup_conv(model,['inp_candidates','embedding_layer','drop','conv1d','global_max_pooling1d_2'],pretrained)
    candidates = model_candidate.predict(concept_padded) # (67782,50)
    logger.info('Formatting pooled mentions and candidates...')
    # from sample import no_cangen_format_x
    from sample import sped_up_format_x
    convoluted_input = sped_up_format_x(mentions,candidates)
    model_sem = _forward_pass_speedup_dot(model,convoluted_input)
    return convoluted_input, model_sem

def _forward_pass_speedup_dot_xDense(original_model,convoluted_x):
    #layerss = ['prediction_layer']
    #d2 = original_model.get_layer(layerss[0])

    pooled_mentions = Input(shape=(convoluted_x[0].shape[1],),dtype='float32', name='pooled_mentions')
    pooled_candidates = Input(shape=(convoluted_x[1].shape[1],),dtype='float32', name='pooled_candidates')
    prediction_layer = layers.dot([pooled_mentions, pooled_candidates], axes=-1, normalize=True, name='cos_sim')
    #prediction_layer = Dense(d2.units, activation=d2.activation,weights=d2.get_weights())(cos_sim)
    
    input_list = [pooled_mentions, pooled_candidates]
    model_part = Model(inputs=input_list, outputs=prediction_layer)
    return model_part

def forward_pass_speedup_shared_encoder_dot_xDense(model,corpus_padded,concept_padded,pretrained):
    '''
    Model to speed up forward pass, used in callback for evaluation
    '''
    model_mention = _forward_pass_speedup_conv(model,['inp_mentions','embedding_layer','drop','conv1d','global_max_pooling1d_1'],pretrained)
    mentions = model_mention.predict(corpus_padded) # (787, 50)
    model_candidate = _forward_pass_speedup_conv(model,['inp_candidates','embedding_layer','drop','conv1d','global_max_pooling1d_2'],pretrained)
    candidates = model_candidate.predict(concept_padded) # (67782,50)
    logger.info('Formatting pooled mentions and candidates...')
    # from sample import no_cangen_format_x
    from sample import sped_up_format_x
    convoluted_input = sped_up_format_x(mentions,candidates)
    model_sem = _forward_pass_speedup_dot_xDense(model,convoluted_input)
    return convoluted_input, model_sem