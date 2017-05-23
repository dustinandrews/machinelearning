# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Reshape, Convolution2D, Flatten, Lambda, MaxPooling2D, Conv2D, UpSampling2D 
from keras import regularizers

import numpy as np
import glob
import tables
from daml.parameters import hyperparameters
import random
from matplotlib import pyplot as plt
from keras import backend as K


#%%
class KAutoEncoder:
    
    def __init__(self, hp: hyperparameters):
        filename = glob.glob('./*/*.h5')[0]
        datafile = tables.open_file(filename)
        self.dfile = datafile
        self.data = datafile.root.earray
        self.hp = hp
        K.clear_session()
        
        
    def create_model(self, output_shape):
        self.output_shape = output_shape
        model = Sequential([
                Dense(self.hp.hidden_dim, activation='relu', input_shape=((1,) + output_shape)),
#               Conv2D(8, (1,1), activation='relu', padding='same',
#                      input_shape=(1,24,80),data_format="channels_first",
#                      ),
               Flatten(),
               Dense(24*80, activation='relu',
                     ),
               Reshape((1,24,80)),
            ])
        model.compile(optimizer=self.hp.optimizer,
                      loss=self.hp.loss,
                      metrics=['accuracy'])
        print([i.name for i in model.layers])
        self.model = model
        return model

    
    def get_samples(self, num_samples: int):
        item_nums = random.sample(range(len(self.data)),num_samples)
        size = [i for i in self.output_shape]
        size.insert(0, 1)
        size.insert(0, num_samples)
        data = np.zeros(tuple(size), dtype=np.float32)
        for i in range(num_samples):
            data[i][0] = self.data[item_nums[i]] / 100
        return data
    
    
    def train_model(self, model = None):
        sample = 2048
        batch = 64
        if model == None:
            self.model = k.create_model((24,80))
        x_train = self.get_samples(sample)
        y_train = x_train
        history = self.model.fit(x_train, y_train, epochs=self.hp.epochs, batch_size=batch)        
        x_test = self.get_samples(128)
        y_test = x_test
        score = self.model.evaluate(x_test, y_test, batch_size=128)
        print("Score {}, Sample Size:{} Batch Size:{}".format(score,batch,sample))
        return history.history
            
    def plot_loss(self, loss, name):                            
        #plotdata["avgloss"] = plotdata["loss"]
        plt.figure(1)    
        plt.subplot(211)
        plt.plot(loss)
        plt.xlabel('Epoch number')
        plt.ylabel('name')
        #plt.yscale('log', basex=10)
        plt.title('Minibatch run vs. ' + name)
        plt.show()        
    
    
    
if __name__ == '__main__':
    hp = hyperparameters(
             hidden_dim=24*80, 
             learning_rate=1e-3, 
             minibatch_size=200,
             epochs=100,
             optimizer='adam',
             loss='mean_squared_error'
             )

    def eval():
        x = k.get_samples(1)
        plt.imshow(x[0][0])
        plt.show()
        y = k.model.predict([x], batch_size=1)      
        plt.imshow(y[0][0])
        plt.show()
    #

    if 'k' in vars() or 'k' in globals():
        k.dfile.close()
        print('closed file')
    k = KAutoEncoder(hp)

    history = k.train_model()
    eval()
    h = []
    num_tests = 1
    for i in range(num_tests):    
        history = k.train_model()
        h.append(history)
    
    run = np.zeros(num_tests)
    for i in range(num_tests):
        run[i] = np.max(h[i]['acc'])

    print("mean: {} median: {} std: {}".format(np.mean(run), np.median(run), np.std(run)))

#    k.plot_loss(history['loss'], 'loss')
#    k.plot_loss(history['acc'], 'acc')
#    eval()
