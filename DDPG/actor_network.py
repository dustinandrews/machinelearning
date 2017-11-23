# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:18:59 2017

@author: dandrews
"""

from keras.models import Sequential
from keras.layers import Dense, Lambda,  BatchNormalization
from keras.initializers import RandomUniform

class ActorNetwork(object):
    optimizer = 'adam'
    loss = 'mse'
    
    def create_actor_network(self, input_shape, output_shape):
        model = Sequential(
                [
                Dense(100, input_shape=input_shape, activation='relu'),
                BatchNormalization(),
                Dense(100, activation='relu'),
                BatchNormalization(),
                Dense(output_shape, 
                      kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003),
                      activation='tanh'
                      ),
                Dense(output_shape, activation='softmax')
                ]                     
                )
        model.compile(optimizer=self.optimizer, loss=self.loss)
        return model
    
if __name__ == '__main__':
    from keras import backend as K
    K.clear_session()
    actor = ActorNetwork()
    model = actor.create_actor_network((1,),1,5)
        
        