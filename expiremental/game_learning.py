# -*- coding: utf-8 -*-
"""
Created on Sun Jun  4 13:45:53 2017

@author: dandrews
"""

import tables
from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Conv2D
from keras.layers import Dropout
from keras import backend as K
import numpy as np
from matplotlib import pyplot as plt

import sys
if r'D:\local\machinelearning' not in sys.path:
    sys.path.append(r'D:\local\machinelearning')
from daml.parameters import hyperparameters

class game_learn:
    
    def __init__(self, hp: hyperparameters):
        data_file = tables.open_file('t-processed.h5')
        self.hp = hp
        self.data_file = data_file
        self.data = data_file.root.data
        self.labels = data_file.root.labels
        self.input_shape = self.data[0].shape
        self.output_shape = self.labels[0].shape
        self.data_len = len(self.labels)
        self.shuffle_order = np.arange(self.data_len)
        np.random.shuffle(self.shuffle_order)
        self.data_index = 0

    def create_model(self):
        K.clear_session()
        model = Sequential()
        model.add(Lambda(lambda x: x/127.5-1.0, input_shape=self.input_shape ))
        model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2), data_format="channels_first", input_shape=self.input_shape))
        model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2), data_format="channels_first"))
        model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2), data_format="channels_first"))
        model.add(Conv2D(64, (3, 3), activation='elu', data_format="channels_first"))
        model.add(Conv2D(64, (3, 3), activation='elu', data_format="channels_first"))
        model.add(Dropout(hp.dropout))
        model.add(Flatten())
        model.add(Dense(100, activation='elu'))
        model.add(Dense(50, activation='elu'))
        model.add(Dense(10, activation='elu'))
        model.add(Dense(self.output_shape[0]))
        model.summary()
        
        model.compile(optimizer=self.hp.optimizer,
              loss=self.hp.loss,
              metrics=['accuracy'])
        
        return model
    
    def get_samples(self, num_samples: int):
        data = np.zeros((num_samples,) + self.input_shape, dtype=np.float32)
        labels = np.zeros((num_samples,) + self.output_shape, dtype=np.float32)
        for i in range(num_samples):
            index = self.data_index % self.data_len
            item = self.shuffle_order[index]
            data[i] = self.data[item]
            labels[i] = self.labels[item]
            index += 1
        return data, labels
    
    
    def train_model(self, model = None):
        sample = 2048
        if model == None:
            self.model = self.create_model()
        x_train, y_train = self.get_samples(sample)
        history = self.model.fit(x_train, y_train, epochs=self.hp.epochs, batch_size=self.hp.minibatch_size, verbose=1) 
        return history.history
            
    def plot_data(self, data, name):                            
        #plotdata["avgloss"] = plotdata["loss"]
        plt.figure(1)    
        plt.subplot(211)
        plt.plot(data)
        plt.xlabel('Epoch number')
        plt.ylabel('name')
        #plt.yscale('log', basex=10)
        plt.title('Minibatch run vs. ' + name)
        plt.show()        
    
    
    
    def cleanup(self):
        self.data_file.close()
    
if __name__ == '__main__':
    hp = hyperparameters()
    hp.epochs = 10
    hp.minibatch_size = 10
    hp.optimizer = 'adam'
    hp.loss = 'mse'
    hp.dropout = 0.20
    
    
    tl = game_learn(hp)
    history = tl.train_model()
    
    