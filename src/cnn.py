

from keras.models import Model
from keras.layers import Input, Embedding, Concatenate, Dense, Conv1D, GlobalMaxPooling1D, Flatten

# needed input: vocabulary, pretrained

for i in range(len(training_data.x)):
    print('training_data.x[{0}]\tshape:{1}\tdtype:{2}'.format(i,training_data.x[i].shape,training_data.x[i].dtype))

#  Remove time distributed and the bugs thrown
inp_mentions = Input(shape=training_data.x[0].shape,dtype='int32', name='inp_mentions')
inp_candidates = Input(shape=training_data.x[1].shape,dtype='int32', name='inp_candidates')
inp_scores = Input(shape=training_data.x[2].shape,dtype='float64', name='inp_scores')

embedding_layer = Embedding(len(vocabulary), pretrained.shape[1], mask_zero=True, trainable=False, weights=[pretrained])
encoded_mentions = embedding_layer(inp_mentions)
encoded_candidates = embedding_layer(inp_candidates)

conv_mentions = Conv1D(filters=50,kernel_size=3,activation='relu')(encoded_mentions) #input_shape=(2000,16,50)

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
# in case they need lists instead of np array
x1list=[mention.tolist() for mention in training_data.x[0]]
inp_mentions_l = Input(shape=training_data.x[0],dtype='int32', name='inp_mentions')
'''


# a layer instance is callable on a tensor, and returns a tensor
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# This creates a model that includes
# the Input layer and three Dense layers
model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(data, labels)  # starts training