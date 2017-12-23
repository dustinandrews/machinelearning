# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:42:23 2017

@author: dandrews
"""

from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Multiply, Input, Flatten
from keras.layers import Concatenate, Add, Conv2D
#import keras

class CriticNetwork(object):
    optimizer = 'adam'
    loss = 'mse'
    merge_layer_size = 100


    def create_critic_network(self, input_shape, action_input_shape, output_shape):
        #print("input_shape {}, action_input_shape {}, output_shape{}".format(input_shape, action_input_shape, output_shape))
        state = Sequential([
                   Conv2D(filters=5, kernel_size=1,input_shape=((input_shape))),
                   Conv2D(filters=5, kernel_size=1),
                  # Flatten(name='state_flatten_1'),
                   Dense(50,activation='relu', name='state_dense_1'),
                   BatchNormalization(name='state_normalization_1'),
                   Dense(100,activation='relu', name='state_dense_2'),
                   BatchNormalization(name='state_normalization_2'),
                   Flatten(name='state_flatten_1'),
                   Dense(self.merge_layer_size, activation='relu', name='state_output_1' )
                   ])

        action =  Sequential([
                Dense(self.merge_layer_size, activation='relu',input_shape=action_input_shape, name='action_dense_1'),
                ])

        #mult =  Add()([action.output,state.output])
        mult = Multiply()([action.output, state.output])

        merged = Dense(50, activation='relu', name='merged_dense')(mult)
        merged = Dense(25, activation='relu', name='critic_dense')(merged)
        merged = Dense(1, activation='tanh', name='critic_out')(merged)
        model = Model(inputs=[state.input, action.input], outputs=merged)
        model.compile(optimizer=self.optimizer, loss=self.loss)
        return state.input, action.input, model


if __name__ == '__main__':
    #%%
    import numpy as np
    from keras import backend as K
    K.clear_session()
    critic = CriticNetwork()
    state_input, action_input, model = critic.create_critic_network((5,5,1), (4,), (1,))
    model.summary()
    #%%
    data = np.ones((1,5,5,1))
    action = np.array([np.arange(4)])
    #%%
    pred = model.predict([data,action],1)
    batch = [np.array(data), np.array(action)]
    print(pred)
    labels = np.ones((1,1))
    model.train_on_batch([data,action],labels)
    pred = model.predict([data,action],1)
    print(pred)
