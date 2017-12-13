# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:18:59 2017

@author: dandrews
"""

from keras.models import Sequential
from keras.layers import Dense, BatchNormalization, Flatten, Conv2D
from keras.initializers import RandomUniform

class ActorNetwork(object):
    optimizer = 'adam'
    loss = 'mse'
    
    def create_actor_network(self, input_shape, output_shape):
        model = Sequential(
                [
                Conv2D(filters=5, kernel_size=1,input_shape=((input_shape))),
               #Flatten(input_shape=input_shape),
                Dense(100,  activation='relu',kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003)),                
                BatchNormalization(),
                Dense(100, activation='relu',kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003)),
                BatchNormalization(),
                Dense(output_shape, 
                      kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003),
                      activation='relu'
                      ),
                Flatten(),
                Dense(output_shape, activation='softmax')
                ]                     
                )
        model.compile(optimizer=self.optimizer, loss=self.loss)
        return model
    
if __name__ == '__main__':
    import numpy as np
    from keras import backend as K
    K.clear_session()
    actor = ActorNetwork()
    model = actor.create_actor_network((10,10,3),5)
    model.summary()
    in_data = np.ones((1, 10,10,3))
    pred = model.predict(in_data)
    print(pred)