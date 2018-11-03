

from keras.models import Model
from keras.layers import Input, Embedding, Concatenate, Dense, Conv1D, GlobalMaxPooling1D, Flatten

# needed input: vocabulary, pretrained, trainging_data

for i in range(len(training_data.x)):
    print('training_data.x[{0}]\tshape:{1}\tdtype:{2}'.format(i,training_data.x[i].shape,training_data.x[i].dtype))

#  Remove the warning thrown (theano)
#  Add Vsem layer (custom layer)
#  Validation data
#  Use the whole training data
#  Custom callback, early stopping?

#  Add Vsem layer (custom layer)
from keras import backend as K
from keras.engine.topology import Layer

class semantic_similarity_layer(Layer):

    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[0][1], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        assert isinstance(x, list)
        a, b = x
        return [K.dot(a, self.kernel) + b, K.mean(b, axis=-1)]

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return [(shape_a[0], self.output_dim), shape_b[:-1]]





'''
x0list=[mention.tolist() for mention in training_data.x[0]]
x1list=[candidate.tolist() for candidate in training_data.x[1]]
x2list=[score.tolist() for score in training_data.x[2]]
x0 = np.array(x0list)
x1 = np.array(x1list)
x2 = np.array(x2list)
inp_mentions = Input(shape=(x0.shape[1],),dtype='int32', name='inp_mentions')
inp_candidates = Input(shape=(x1.shape[1],),dtype='int32', name='inp_candidates')
inp_scores = Input(shape=(x2.shape[1],),dtype='float64', name='inp_scores')

training_data.y = np.array(training_data.y)
'''

inp_mentions = Input(shape=(training_data.x[0].shape[1],),dtype='int32', name='inp_mentions')
inp_candidates = Input(shape=(training_data.x[1].shape[1],),dtype='int32', name='inp_candidates')
inp_scores = Input(shape=(training_data.x[2].shape[1],),dtype='float64', name='inp_scores')

embedding_layer = Embedding(len(vocabulary), pretrained.shape[1], mask_zero=False, trainable=False, weights=[pretrained])
encoded_mentions = embedding_layer(inp_mentions)
encoded_candidates = embedding_layer(inp_candidates)

conv_mentions = Conv1D(filters=50,kernel_size=3,activation='relu')(encoded_mentions) #input_shape=(2000,16,50)
conv_candidates = Conv1D(filters=50,kernel_size=3,activation='relu')(encoded_candidates) #input_shape=(2000,16,50)
pooled_mentions = GlobalMaxPooling1D()(conv_mentions)
pooled_candidates = GlobalMaxPooling1D()(conv_candidates)

join_layer = Concatenate()([pooled_mentions,pooled_candidates,inp_scores])
hidden_layer = Dense(64, activation='relu')(join_layer)
prediction_layer = Dense(1,activation='sigmoid')(join_layer)

model = Model(inputs=[inp_mentions,inp_candidates,inp_scores], outputs=prediction_layer)
model.compile(optimizer='adadelta', loss='binary_crossentropy')

hist = model.fit([x0,x1,x2], training_data.y, epochs=50, batch_size=20)
# WARNING (theano.tensor.blas): We did not found a dynamic library into the library_dir of the library we use for blas. If you use ATLAS, make sure to compile it with dynamics library.


'''
y=np.array(training_data.y)
model.fit([x0,x1,x2], y, epochs=50, batch_size=20)
model.fit([x0,x1,x2], training_data.y, epochs=50, batch_size=20)
#model.fit(training_data.x, training_data.y, epochs=50, batch_size=20)
'''
'''
>>> model.summary()
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
inp_mentions (InputLayer)       (None, 16)           0                                            
__________________________________________________________________________________________________
inp_candidates (InputLayer)     (None, 17)           0                                            
__________________________________________________________________________________________________
embedding_3 (Embedding)         multiple             2500100     inp_mentions[0][0]               
                                                                 inp_candidates[0][0]             
__________________________________________________________________________________________________
conv1d_5 (Conv1D)               (None, 14, 50)       7550        embedding_3[0][0]                
__________________________________________________________________________________________________
conv1d_6 (Conv1D)               (None, 15, 50)       7550        embedding_3[1][0]                
__________________________________________________________________________________________________
global_max_pooling1d_5 (GlobalM (None, 50)           0           conv1d_5[0][0]                   
__________________________________________________________________________________________________
global_max_pooling1d_6 (GlobalM (None, 50)           0           conv1d_6[0][0]                   
__________________________________________________________________________________________________
inp_scores (InputLayer)         (None, 1)            0                                            
__________________________________________________________________________________________________
concatenate_3 (Concatenate)     (None, 101)          0           global_max_pooling1d_5[0][0]     
                                                                 global_max_pooling1d_6[0][0]     
                                                                 inp_scores[0][0]                 
__________________________________________________________________________________________________
dense_6 (Dense)                 (None, 1)            102         concatenate_3[0][0]              
==================================================================================================
Total params: 2,515,302
Trainable params: 15,202
Non-trainable params: 2,500,100
__________________________________________________________________________________________________

'''

'''
inp_mentions = Input(shape=training_data.x[0].shape,dtype='int32', name='inp_mentions')
inp_candidates = Input(shape=training_data.x[1].shape,dtype='int32', name='inp_candidates')
inp_scores = Input(shape=training_data.x[2].shape,dtype='float64', name='inp_scores')

embedding_layer = Embedding(len(vocabulary), pretrained.shape[1], mask_zero=True, trainable=False, weights=[pretrained])
encoded_mentions = embedding_layer(inp_mentions)
encoded_candidates = embedding_layer(inp_candidates)

conv_mentions = Conv1D(input_shape=(16,50),filters=50,kernel_size=3,activation='relu')(encoded_mentions) #input_shape=(2000,16,50)

conv_mentions = TimeDistributed(Conv1D(filters=50,kernel_size=3,activation='relu'))(encoded_mentions) #input_shape=(2000,16,50)
conv_candidates = TimeDistributed(Conv1D(filters=50,kernel_size=3,activation='relu'))(encoded_candidates) #input_shape=(2000,17,50)

pooled_mentions = TimeDistributed(GlobalMaxPooling1D())(conv_mentions)
pooled_candidates = TimeDistributed(GlobalMaxPooling1D())(conv_candidates)

#expt_layer_mentions = Flatten()(pooled_mentions)
#TypeError: Layer flatten_1 does not support masking, but was passed an input_mask: Elemwise{neq,no_inplace}.0

join_layer = Concatenate()([pooled_mentions,pooled_candidates,inp_scores])

hidden_layer = TimeDistributed(Dense(64, activation='relu'))(join_layer)
prediction_layer = TimeDistributed(Dense(1,activation='sigmoid'))(join_layer)

model = Model(inputs=[inp_mentions,inp_candidates,inp_scores], outputs=prediction_layer)
model.compile(optimizer='adadelta', loss='binary_crossentropy',sample_weight_mode='temporal')

model.fit(training_data.x, training_data.y, epochs=50, batch_size=20)
#ValueError: Error when checking input: expected inp_mentions to have 3 dimensions, but got array with shape (2000, 16)
#model.fit([training_data.x[0],training_data.x[1],training_data.x[2]], training_data.y, epochs=50, batch_size=20)
'''