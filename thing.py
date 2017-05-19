# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Reshape
import numpy as np
import glob
import tables
from daml.parameters import hyperparameters
import random
from matplotlib import pyplot as plt

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
            Dense(output_shape * 10, input_shape=(output_shape,)),
            Activation('relu'),
            Dense(output_shape),
            Activation('linear')
            ])
        model.compile(optimizer=self.hp.optimizer,
                      loss=self.hp.loss,
                      metrics=['accuracy'])
        print([i.name for i in model.layers])
        self.model = model
        return model

    
    def get_samples(self, num_samples: int):
        item_nums = random.sample(range(len(self.data)),num_samples)
        size = [self.output_shape]
        size.insert(0, num_samples)
        data = np.zeros(tuple(size), dtype=np.float32)
        for i in range(num_samples):
            data[i] = self.data[item_nums[i]].flatten() / 100
        return data
    
    
    def train_model(self):
        x_train = self.get_samples(1000)
        y_train = x_train
        history = self.model.fit(x_train, y_train, epochs=100, batch_size=128)
        
        x_test = self.get_samples(128)
        y_test = x_test
        score = self.model.evaluate(x_test, y_test, batch_size=128)
        print(score)
        return history.history
            
        
    
    
    
if __name__ == '__main__':
    hp = hyperparameters(
             hidden_dim=24*80, 
             learning_rate=1e-4, 
             minibatch_size=200,
             epochs=10,
             optimizer='adagrad',
             loss='mean_squared_error'
             )

    def eval():
        x = k.data[1000]
        plt.imshow(x)
        plt.show()
        x = x.reshape(1,1920)
        y = k.model.predict(x, batch_size=1)
        y = y.reshape(24, 80).astype(np.int)
        plt.imshow(y)
        plt.show()
    #
    
    k = KAutoEncoder(hp)
    k.model = k.create_model(24*80)

    eval()    

    history = k.train_model()
    
    # plot history['loss']
    
    eval()
