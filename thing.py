# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
import numpy as np
import glob
import tables
from parameters import hyperparameters
import random

#%%
class KAutoEncoder:
    
    def __init__(self, hp: hyperparameters):
        filename = glob.glob('./*/*.h5')[0]
        datafile = tables.open_file(filename)
        self.data = datafile.root.earray
        self.hp = hp
        
        
    def create_model(self, output_shape):
        self.output_shape = output_shape
        model = Sequential([
            Embedding(1, 24 * 80),
            Dense(self.hp.hidden_dim),
            Activation('relu'),
            Dense(output_shape)            
            ])
        model.compile(optimizer=self.hp.optimizer,
                      loss=self.hp.loss,
                      metrics=['accuracy'])
        return model

    
    def get_samples(self, num_samples: int):
        item_nums = random.sample(range(len(self.data)),num_samples)
        size = [self.output_shape]
        size.insert(0, num_samples)
        data = np.zeros(tuple(size), dtype=np.float32)
        for i in range(num_samples):
            data[i] = self.data[item_nums[i]].flatten()
        return data
    
    
    def train_model(self):
        model = self.create_model((24 * 80))
        x_train = self.get_samples(1000)
        y_train = x_train
        model.fit(x_train, y_train, epochs=10, batch_size=128)
        x_test = self.get_samples(128)
        y_test = x_test
        score = model.evaluate(x_test, y_test, batch_size=128)
        print(score)
            
        
    
    
    
if __name__ == '__main__':
    hp = hyperparameters(
             hidden_dim=24*80, 
             learning_rate=1e-4, 
             minibatch_size=100,
             epochs=10,
             optimizer='adagrad',
             loss='mean_squared_error'
             )
    k = KAutoEncoder(hp)   
    