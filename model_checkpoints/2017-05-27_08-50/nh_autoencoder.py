# -*- coding: utf-8 -*-
"""
Spyder Editor

Autoencoder for Nethack TTY frames that
have been converted to (ord(c)-32)/100 data maps
Converges well but accuracy is below 50%
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
import random
from matplotlib import pyplot as plt
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.constraints import maxnorm

import sys
mlpath = r'C:\local\machinelearning'
if not mlpath in sys.path:
    sys.path.append(mlpath)
from daml.parameters import hyperparameters


#%%
class KAutoEncoder:
    
    def __init__(self, hp: hyperparameters):
        filename = glob.glob(mlpath + '/tty.h5')[0]
        datafile = tables.open_file(filename)
        self.dfile = datafile
        self.data = datafile.root.earray
        self.hp = hp
        self.input_shape = (24,80)
        self.zero_line = np.zeros(self.input_shape[1])
       
    def create_model(self):
        K.clear_session()
        model = Sequential([
               Dropout(0.15, input_shape=(1,24,80)),
               Conv2D(8, (1,1), activation='relu', padding='valid',
                      input_shape=(1,24,80),data_format="channels_first",
                      ),
               MaxPooling2D(pool_size=(2, 2)),
               Conv2D(8, (3,1), activation='relu', padding='valid',
                      input_shape=(1,24,80),data_format="channels_first",
                      ),
               Conv2D(8, (3,1), activation='relu', padding='valid',
                      input_shape=(1,24,80),data_format="channels_first",
                      ), 
               Dropout(0.15),
               MaxPooling2D(pool_size=(2, 2),data_format="channels_first"),            
               Flatten(),
#               Dense(self.hp.hidden_dim, activation='relu'),
               Dense(self.hp.hidden_dim, activation='relu',kernel_constraint=maxnorm(3)),
               Dense(24*80, activation='relu',kernel_constraint=maxnorm(3)),
               Reshape((1,24,80)),
            ])
               
        model.compile(optimizer=self.hp.optimizer,
                      loss=self.hp.loss,
                      metrics=['mse','accuracy'])
        
        K.set_value(model.optimizer.lr, self.hp.learning_rate)
        
        print([i.name for i in model.layers])
        self.model = model
        return model

    
    def get_samples(self, num_samples: int):
        item_nums = random.sample(range(len(self.data)),num_samples)
        size = [i for i in self.input_shape]
        size.insert(0, 1)
        size.insert(0, num_samples)
        data = np.zeros(tuple(size), dtype=np.float32)
        
        for i in range(num_samples):
            d = self.data[item_nums[i]]
            # Move some items around to group them better
            d[d == 13] = 92 # convert - to | to unify walls
            d[d == 32] = 100 # convert @ to maxval
            d[d == 3] = 14 # convert # to .
            data[i][0] = d / 100
            #clear non map lines
            data[i][0][0] = self.zero_line
            data[i][0][22] = self.zero_line
            data[i][0][23] = self.zero_line 
        return data
    
    
    def train_model(self, model = None):
#        filepath="weights-best.hdf5"
#        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='max')
#        callbacks_list = [checkpoint]
        sample = self.hp.minibatch_size * 10
        if model == None:
            self.model = self.create_model()
        x_train = self.get_samples(sample)
        y_train = x_train
        print(self.hp.epochs)
        history = self.model.fit(x_train, y_train, epochs=self.hp.epochs, batch_size=self.hp.minibatch_size, verbose=1) 
#        history = self.model.fit(x_train, y_train, epochs=self.hp.epochs, batch_size=batch, callbacks=callbacks_list)        
        x_test = self.get_samples(self.hp.minibatch_size)
        y_test = x_test
        score = self.model.evaluate(x_test, y_test, batch_size=self.hp.minibatch_size)
        print("Test Score(loss, mse, acc {}".format(score))
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
    
    
    def plot_items(self, args, name="data"):                            
        plt.figure(1, dpi=150)
        plt.subplot(211)
        lines = []
        for k,v in args.items():
            lines += plt.plot(v, label=k)
        labels = [l.get_label() for l in lines]
        plt.legend(lines, labels)
        plt.xlabel('Epoch number')
        plt.ylabel('value')
        plt.yscale('log', basex=10)
        plt.title(name)
        plt.show() 

#%% 
if __name__ == "__main__":   
    def eval():
        x = k_auto.get_samples(1)
        plt.matshow(x[0][0])
        plt.show()
        y = k_auto.model.predict([x], batch_size=1)      
        plt.matshow(y[0][0])
        plt.show()
        return x,y
    
    def show_as_text(in_sample: np.array):
        sample = in_sample.copy().reshape((24,80))
        
        sample *= 100
        sample += 32
        
        for i in sample:
            line =[]
            for j in i:
                line.append(chr(int(j)))
            print("".join(line))
#%%
    hp = hyperparameters(
             hidden_dim=int(24*80*1.5), 
             learning_rate=1e-3, 
             minibatch_size=128,
             epochs=20,
             optimizer='rmsprop',
             #loss=loss
             )
      
    record = {}    
    for loss in ('mean_squared_logarithmic_error',):
        hp.loss = loss
        if 'k_auto' in vars() or 'k_auto' in globals():
            k_auto.dfile.close()
            print('closed file')
        k_auto = KAutoEncoder(hp)
 #%%   
    #    history = k_auto.train_model()
    #    eval()
        h = []
        num_tests = 1
        for i in range(num_tests):    
            history = k_auto.train_model()
            h.append(history)
        
        run = {}
        for i in range(num_tests):
            run[i] = (np.max(h[i]['acc']),np.max(h[i]['loss']),np.max(h[i]['mean_squared_error']))
    
#        print("mean: {} median: {} std: {}".format(np.mean(run), np.median(run), np.std(run)))
        print([("{}: {}".format(k, hp.__dict__[k])) for k in hp.__dict__])
        record[loss] = run
        eval()
        k_auto.plot_items(h[0])
    print(record)
#    k_auto.plot_loss(history['loss'], 'loss')
#    k_auto.plot_loss(history['acc'], 'acc')
#    eval()
