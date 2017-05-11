# -*- coding: utf-8 -*-
"""
Created on Sun May  7 12:22:02 2017

@author: Dustin
"""
from scipy.sparse import coo_matrix
from ttyrec import ttyparse
import cntk as C
import numpy as np
import tables
import glob

class CategoryAutoEncoder:
    _ttyrec_file = "ttyrec/recordings/stth/2010-02-07.12_39_37.ttyrec"
    _num_categories = 100
    
    def __init__(self):
        filename = glob.glob('./*/*/*/*.hf5')[0]
        datafile = tables.open_file(filename)
        self.data = datafile.root.carray
        self.current_input = self.data[0][0]
    
    
    def create_model(self, input_dim, output_dim, hidden_dim, feature_input):    
        """
        Create a model with the layers library.
        """        
#        cmap = 20
#        num_channels = 1
        
        netout = C.layers.Sequential([
                 #C.layers.Embedding(input_dim),
                 C.layers.Dense(input_dim, activation=C.ops.sigmoid),
                 C.layers.Dense(output_dim, activation=C.ops.sigmoid)
                ])(feature_input)
        
#        c1 = C.layers.Convolution2D((3,3), cmap, strides=2, reduction_rank=0)(feature_input)
#        p1 = C.layers.MaxPooling( (3,3), (2,2))(c1)
#        d1 = C.layers.Dense((20,5,19))(p1)
#        u1 = C.layers.MaxUnpooling((3,3), (2,2))(p1, c1)
#        #C.layers.Dense(output_dim)
#        d1 = C.layers.ConvolutionTranspose2D((3,3), num_channels, bias=False, init=C.glorot_uniform(0.001))(u1)
#        netout = C.layers.Dense(output_dim)(d1)        
        return(netout)
    
    
    def create_model_conv(self, input_dim, output_dim, hidden_dim, feature_input):    
        """
        Create a model with the layers library.
        """        
        cmap = 20
        num_channels = 1
        
        c1 = C.layers.Convolution2D((3,3), cmap, strides=2,activation=C.ops.sigmoid, pad=True, reduction_rank=0)(feature_input)
        #d1 = C.layers.Dense((20,12,40), activation=C.ops.sigmoid)(c1)
        p1 = C.layers.MaxPooling( (3,3), (2,2))(c1)        
        u1 = C.layers.MaxUnpooling((3,3), (2,2))(p1, c1)        
        d1 = C.layers.ConvolutionTranspose2D((3,3), num_channels, pad=True, bias=False, init=C.glorot_uniform(0.001))(u1)
        netout = C.layers.Dense(output_dim, activation=C.ops.sigmoid)(d1)        
        return(netout)
    
    def create_sparse_model(self, input_dim, output_dim, hidden_dim, feature_input):
        my_model = C.layers.Sequential([
                C.layers.Embedding(self._num_categories),
                C.layers.Dense(hidden_dim)
                ])
        netout = my_model(feature_input)
        return netout
    
    
    def get_next_data_2d_one_hot(self, num_records):
        self.shape = (1, self.ttyparse.metadata.lines, self.ttyparse.metadata.collumns)
        batch = []
        for _ in range(num_records):
            next_in = np.array(self.ttyparse.get_next_render_flat(), dtype=np.int32)
            #next_in = next_in * (1/self._num_categories)
            next_in = next_in.reshape(self.shape)
            small_sample = next_in[:,-15:-5,-30:-20]
            one_hot = self.convert_to_one_hot(small_sample)
            batch.append(one_hot)
        np_batch = np.array(batch, dtype=np.float32)
        return np_batch

    def get_next_data(self, num_records):
        self.shape = (self.ttyparse.metadata.lines, self.ttyparse.metadata.collumns)
        batch = []
        for _ in range(num_records):
            next_in = np.array(self.ttyparse.get_next_render_flat(), dtype=np.int32)
            #next_in = next_in * (1/self._num_categories)
            next_in = next_in.reshape(self.shape)
            small_sample = next_in[-15:-5,-30:-20]
            one_hot = self.convert_to_one_hot(small_sample)
            batch.append(one_hot)            
        np_batch = np.array(batch, dtype=np.float32)
        return np_batch
    
    def convert_to_one_hot(self, np_arr):
        n_values = self._num_categories + 1
        ret_array = np.eye(n_values)[np_arr]
        return ret_array

    def moving_average(self, a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    
    def one_hot_to_indexs(self, i):
        return [[np.argmax(i) for i in line] for line in x[0]]
    
    def custom_loss(self, prediction, target, name=''):
        return (prediction == target).sum()

#%%    
if __name__ == '__main__':
    """
    Run from __main__ to allow easier interaction with immediate window

    Hyperparameters
    """
    self = CategoryAutoEncoder()
    input_dim = self.current_input.shape
    output_dim = self.current_input.shape
    hidden_dim = self.current_input.shape
    learning_rate = 1e-3
    minibatch_size = 2
    epoch_size = 10
    batch_size = 100
    
    """
    Input and output shapes
    """
#    feature = C.input((input_dim), np.float32)
#    label = C.input((output_dim), np.float32)
    feature = C.input((input_dim), np.float32)
    label = C.input((output_dim), np.float32)
    
    #netout = self.create_model(input_dim, output_dim, hidden_dim, feature)
    netout = self.create_model(input_dim, output_dim, hidden_dim, feature)
    #loss = C.cross_entropy_with_softmax(netout, feature)
    #loss = C.squared_error(netout, feature)    
    loss = C.cross_entropy_with_softmax(netout, feature, axis=0)
    evaluation = C.squared_error(netout, feature)
    lr_per_minibatch= C.learning_rate_schedule(learning_rate, C.UnitType.minibatch)
    
    #learner = C.sgd(netout.parameters, lr=lr_per_minibatch)
    #learner = C.sgd(netout.parameters, lr=lr_per_minibatch, l2_regularization_weight=0.001)
    schedule = C.momentum_schedule(learning_rate)
    learner = C.adam(netout.parameters, C.learning_rate_schedule(learning_rate, C.UnitType.minibatch), momentum=schedule, l2_regularization_weight=0.001)
    
    progress_printer = C.logging.ProgressPrinter(epoch_size * batch_size)
    
    trainer = C.Trainer(netout, (loss, evaluation), learner, progress_printer)
    
    plotdata = {"loss":[]}
    for epoch in range(epoch_size):
        for i in range(batch_size):
            d = self.get_next_data(minibatch_size)
            data = {feature : d, label : d}
            """
            # This is how to get the Numpy typed data from the reader
            ldata = data[label].asarray()
            fdata = data[feature].asarray()
            """
            trainer.train_minibatch(data)
            loss = trainer.previous_minibatch_loss_average
            if not (loss == "NA"):
                plotdata["loss"].append(loss)       
            if np.abs(trainer.previous_minibatch_loss_average) < 0.0015: #stop once the model is good.
                break
#%%
        trainer.summarize_training_progress()
#    test_data = training_reader.next_minibatch(minibatch_size, input_map = input_map)
#    avg_error = trainer.test_minibatch(test_data)
#    print("Error rate on an unseen minibatch %f" % avg_error)
    
#%%
    import matplotlib.pyplot as plt
    if len(plotdata["loss"]) > 100:
        plotdata["avgloss"] = self.moving_average(plotdata["loss"], 100)
    else:
        plotdata["avgloss"] = plotdata["loss"]
    #plotdata["avgloss"] = plotdata["loss"]
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["avgloss"])
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss')
    plt.show()
#%%    
    def renderSample(x):
        lines = []
        xy = [[np.argmax(i) for i in line] for line in x ]
        xy = np.array(xy) + 32
        for line in xy:
            new_line = ""
            for i in line:
                if i < 32:
                    i = ord("ñ")
                new_line += chr(i)            
            #print("'" + new_line + "'")
            lines.append(new_line)
        return lines


    
#%%

    x = self.get_next_data(1)[0]
    y = netout.eval({feature: x})

    rx = renderSample(x)
    print()
    ry = renderSample(y[0])
    
    for i in range(len(rx)):
        diff = ""
        for x in range(len(rx[i])):
            if rx[i][x] == ry[i][x]:
                diff += ry[i][x]
            else:
                diff += " "
                
        print("'{}'   '{}'   '{}'".format(rx[i], ry[i], diff))
    

#%%

    def iter_data():
        maxsize = self.ttyparse.metadata.num_frames
        read = 0
        while(True):
            read += 1
            data = self.get_next_data(1)[0]
            if read < maxsize:
                yield data
            else:
                break
            
        
    #sample = savedata(10)               
            