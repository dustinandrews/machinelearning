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
    
    def create_critic_network(self, input_shape, output_shape, action_input_shape):
        state = Sequential([
                   Flatten(input_shape=input_shape),
                   Dense(400,activation='relu'),
                   BatchNormalization(),
                   Dense(300, activation='relu')
                   ])

        action =  Sequential([
                Dense(300, activation='relu',input_shape=action_input_shape),
                ])

        mult =  Add()([action.output,state.output])
        merged = Dense(1)(mult)
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
    print(pred)
    