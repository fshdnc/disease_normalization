#!/usr/bin/env python3
# coding: utf8

from keras.models import Model
from keras.layers import Input, Embedding, Concatenate, Dense, Conv1D, GlobalMaxPooling1D, Flatten

import logging
logger = logging.getLogger(__name__)

# needed input: vocabulary, pretrained, trainging_data

def print_input(training_data):
    for i in range(len(training_data.x)):
        logger.info('training_data.x[{0}]\tshape:{1}\tdtype:{2}'.format(i,training_data.x[i].shape,training_data.x[i].dtype))

#  Remove the warning thrown (theano)
#  Add Vsem layer (custom layer)
#  Validation data
#  Use the whole training data
#  Custom callback, early stopping?

#  Add Vsem layer (custom layer)
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

        self.M = self.add_weight(name='M',
                                 shape=(shape_m[1], shape_c[1]),
                                 initializer='uniform',
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

def build_model(conf,training_data,vocabulary,pretrained):
    inp_mentions = Input(shape=(training_data.x[0].shape[1],),dtype='int32', name='inp_mentions')
    inp_candidates = Input(shape=(training_data.x[1].shape[1],),dtype='int32', name='inp_candidates')
    if int(conf['candidate']['use']):
        inp_scores = Input(shape=(training_data.x[2].shape[1],),dtype='float64', name='inp_scores')
    if conf.getint('embedding','elmo'):
        inp_mentions_elmo = Input(shape=(training_data.x[-2].shape[1],),dtype='float32', name='inp_mentions_elmo')
        inp_candidates_elmo = Input(shape=(training_data.x[-1].shape[1],),dtype='float32', name='inp_candidates_elmo')

    embedding_layer = Embedding(len(vocabulary), pretrained.shape[1], mask_zero=False, trainable=False, weights=[pretrained])
    encoded_mentions = embedding_layer(inp_mentions)
    encoded_candidates = embedding_layer(inp_candidates)

    conv_mentions = Conv1D(filters=50,kernel_size=3,activation='relu')(encoded_mentions) #input_shape=(2000,16,50)
    conv_candidates = Conv1D(filters=50,kernel_size=3,activation='relu')(encoded_candidates) #input_shape=(2000,16,50)
    pooled_mentions = GlobalMaxPooling1D()(conv_mentions)
    pooled_candidates = GlobalMaxPooling1D()(conv_candidates)
    if conf.getint('embedding','elmo'):
        conv_mentions_elmo = Conv1D(filters=50,kernel_size=3,activation='relu')(inp_mentions_elmo)
        conv_candidates_elmo = Conv1D(filters=50,kernel_size=3,activation='relu')(inp_candidates_elmo)
        pooled_mentions_elmo = GlobalMaxPooling1D()(conv_mentions_elmo)
        pooled_candidates_elmo = GlobalMaxPooling1D()(conv_candidates_elmo)
        v_sem_elmo = semantic_similarity_layer()([pooled_mentions_elmo,pooled_candidates_elmo])

    v_sem = semantic_similarity_layer()([pooled_mentions,pooled_candidates])

    # list of layers for concatenation
    concatenate_list = [pooled_mentions,pooled_candidates,v_sem]
    if int(conf['candidate']['use']):
        concatenate_list.append(inp_scores)
    if conf.getint('embedding','elmo'):
        concatenate_list.extend([pooled_mentions_elmo,pooled_candidates_elmo,v_sem_elmo])

    join_layer = Concatenate()(concatenate_list)
    hidden_layer = Dense(64, activation='relu')(join_layer)
    if int(conf['settings']['imp_tr']):
        from keras.layers import Activation
        prediction_layer = Activation('sigmoid')(hidden_layer)
    else:
        prediction_layer = Dense(1,activation='sigmoid')(hidden_layer)  

    # list of input layers
    input_list = [inp_mentions,inp_candidates]
    if int(conf['candidate']['use']):
        input_list.append(inp_scores)
    if conf.getint('embedding','elmo'):
        input_list.extend([inp_mentions_elmo,inp_candidates_elmo])

    model = Model(inputs=input_list, outputs=prediction_layer)
    from keras import optimizers
    #adagrad = optimizers.Adagrad(lr=0.001, epsilon=None, decay=0.0)
    model.compile(optimizer='adadelta', loss='binary_crossentropy')

    return model

'''
>>> model.summary()
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
inp_mentions (InputLayer)       (None, 16)           0                                            
__________________________________________________________________________________________________
inp_candidates (InputLayer)     (None, 27)           0                                            
__________________________________________________________________________________________________
embedding_1 (Embedding)         multiple             2500100     inp_mentions[0][0]               
                                                                 inp_candidates[0][0]             
__________________________________________________________________________________________________
conv1d_1 (Conv1D)               (None, 14, 50)       7550        embedding_1[0][0]                
__________________________________________________________________________________________________
conv1d_2 (Conv1D)               (None, 25, 50)       7550        embedding_1[1][0]                
__________________________________________________________________________________________________
global_max_pooling1d_1 (GlobalM (None, 50)           0           conv1d_1[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_2 (GlobalM (None, 50)           0           conv1d_2[0][0]                   
__________________________________________________________________________________________________
inp_scores (InputLayer)         (None, 1)            0                                            
__________________________________________________________________________________________________
semantic_similarity_layer_4 (se (None, 1)            2500        global_max_pooling1d_1[0][0]     
                                                                 global_max_pooling1d_2[0][0]     
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 102)          0           global_max_pooling1d_1[0][0]     
                                                                 global_max_pooling1d_2[0][0]     
                                                                 inp_scores[0][0]                 
                                                                 semantic_similarity_layer_4[0][0]
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 1)            103         concatenate_1[0][0]              
==================================================================================================
Total params: 2,517,803
Trainable params: 17,703
Non-trainable params: 2,500,100
__________________________________________________________________________________________________

'''
