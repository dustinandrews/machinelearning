# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Reshape, Convolution2D
from keras.layers import Flatten, Lambda, MaxPooling2D, Conv2D, UpSampling2D
from keras.layers import Dropout
from keras.optimizers import Adam
from keras import regularizers

import numpy as np
import glob
import tables
from daml.parameters import hyperparameters
import random
from matplotlib import pyplot as plt
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.constraints import maxnorm

#%%
class KAutoEncoder:
    
    def __init__(self, hp: hyperparameters):
        filename = glob.glob('./*/*.h5')[0]
        datafile = tables.open_file(filename)
        self.dfile = datafile
        self.data = datafile.root.earray
        self.hp = hp
        
        
        
        
    def create_model(self, output_shape):
        K.clear_session()
        self.output_shape = output_shape
        model = Sequential([
               Dropout(0.15, input_shape=(1,24,80)),
               Conv2D(8, (1,1), activation='relu', padding='valid',
                      input_shape=(1,24,80),data_format="channels_first",
                      ),
               Dropout(0.15),
               Flatten(),
#               Dense(self.hp.hidden_dim, activation='relu'),
               Dense(self.hp.hidden_dim, activation='relu',kernel_constraint=maxnorm(3)),
               Dropout(0.15),
               Dense(24*80, activation='relu',kernel_constraint=maxnorm(3)),
               Reshape((1,24,80)),
            ])
               
        model.compile(optimizer=self.hp.optimizer,
                      loss=self.hp.loss,
                      metrics=['accuracy'])
        print(model.summary())
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
#        filepath="weights-best.hdf5"
#        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
#        callbacks_list = [checkpoint]
        sample = 2048
        if model == None:
            self.model = k.create_model((24,80))
        x_train = self.get_samples(sample)
        y_train = x_train
        history = self.model.fit(x_train, y_train, epochs=self.hp.epochs, batch_size=self.hp.minibatch_size, verbose=0) 
#        history = self.model.fit(x_train, y_train, epochs=self.hp.epochs, batch_size=batch, callbacks=callbacks_list)        
#        x_test = self.get_samples(128)
#        y_test = x_test
#        score = self.model.evaluate(x_test, y_test, batch_size=128)
#        print("Score {}, Sample Size:{} Batch Size:{}".format(score,self.hp.minibatch_size,sample))
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
             hidden_dim=int(24*80*1.5), 
             learning_rate=1e-4, 
             minibatch_size=200,
             epochs=75,
             optimizer='rmsprop',
             loss='logcosh'
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

#    history = k.train_model()
#    eval()
    h = []
    num_tests = 3
    for i in range(num_tests):    
        history = k.train_model()
        h.append(history)
    
    run = np.zeros(num_tests)
    for i in range(num_tests):
        run[i] = np.max(h[i]['acc'])

    print("mean: {} median: {} std: {}".format(np.mean(run), np.median(run), np.std(run)))
    print([("{}: {}".format(k, hp.__dict__[k])) for k in hp.__dict__])
#    record[opt] = np.median(run)
    
#    k.plot_loss(history['loss'], 'loss')
#    k.plot_loss(history['acc'], 'acc')
#    eval()
