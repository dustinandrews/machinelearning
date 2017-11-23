# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:42:23 2017

@author: dandrews
"""

from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Multiply, Input

class CriticNetwork(object):
    optimizer = 'adam'
    loss = 'mse'
    
    def create_critic_network(self, input_shape, output_shape, action_input_shape):
        state_input = Input(shape=input_shape)
        state = Sequential([
                   Dense(400,activation='relu', input_shape=input_shape),
                   BatchNormalization(),
                   Dense(300, activation='relu')
                   ])(state_input)
        
        action_input = Input(shape=action_input_shape)
        action =  Sequential([
                Dense(300, activation='relu',input_shape=action_input_shape)
                ])(action_input)

        
        merged =  Multiply()([state, action])   

        model = Model(inputs=[state_input, action_input], outputs=merged)
        model.compile(optimizer=self.optimizer, loss=self.loss)        
        return model
        
        
if __name__ == '__main__':
    from keras import backend as K
    K.clear_session()
    critic = CriticNetwork()
    model = critic.create_critic_network((1,), 1, (1,))