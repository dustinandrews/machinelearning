# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:42:23 2017

@author: dandrews
"""

from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Multiply, Input, Flatten, Concatenate, Add

class CriticNetwork(object):
    optimizer = 'adam'
    loss = 'mse'
    merge_layer_size = 100
    
    def create_critic_network(self, input_shape, output_shape, action_input_shape):
        state = Sequential([
                   Flatten(input_shape=input_shape, name='state_flatten_1'),
                   Dense(100,activation='relu', name='state_dense_1'),
                   BatchNormalization(name='state_normalization_1'),
                   Dense(200,activation='relu', name='state_dense_2'),
                   BatchNormalization(name='state_normalization_2'),
                   Dense(self.merge_layer_size, activation='relu', name='state_output_1' )
                   ])

        action =  Sequential([
                Dense(self.merge_layer_size, activation='relu',input_shape=action_input_shape, name='action_dense_1'),
                ])

        #mult =  Add()([action.output,state.output])
        mult = Multiply()([action.output, state.output])
        
        merged = Dense(100, activation='relu', name='merged_dense')(mult)
        merged = Dense(50, activation='relu', name='critic_lense')(merged)
        merged = Dense(1, activation='sigmoid', name='critic_out')(merged)
        model = Model(inputs=[state.input, action.input], outputs=merged)
        model.compile(optimizer=self.optimizer, loss=self.loss)        
        return model
        
        
if __name__ == '__main__':
    #%%
    import numpy as np
    from keras import backend as K
    K.clear_session()
    critic = CriticNetwork()
    model = critic.create_critic_network((10,10), 1, (1,))
    #%%
    data = np.ones((1,10,10))
    action = np.array([1])
    #%%
    pred = model.predict([data,action],1)
    batch = [np.array(data), np.array(action)]
    print(pred)
    labels = np.ones((1,1))
    model.train_on_batch([data,action],labels)
    pred = model.predict([data,action],1)
    print(pred)
    