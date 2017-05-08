# -*- coding: utf-8 -*-
"""
Created on Sun May  7 12:22:02 2017

@author: Dustin
"""
from scipy.sparse import coo_matrix
from ttyrec import ttyparse
import cntk as C
import numpy as np

class CategoryAutoEncoder:
    _ttyrec_file = "ttyrec/recordings/stth/2010-02-07.12_39_37.ttyrec"
    _num_categories = 89
    
    def __init__(self):
        self.ttyparse = ttyparse.TtyParse(self._ttyrec_file)
        self.current_input = self.get_next_data(1)[0]
    
    def create_model(self, input_dim, output_dim, hidden_dim, feature_input):    
        """
        Create a model with the layers library.
        """        
        cmap = 2
        num_channels = 1
        
        c1 = C.layers.Convolution2D((3,3), cmap, strides=2,activation=C.ops.relu, pad=True, reduction_rank=0)(feature_input)
        p1 = C.layers.MaxPooling( (3,3), (2,2))(c1)
        u1 = C.layers.MaxUnpooling((3,3), (2,2))(p1, c1)
        #C.layers.Dense(output_dim)
        d1 = C.layers.ConvolutionTranspose2D((3,3), num_channels, pad=True, bias=False, init=C.glorot_uniform(0.001))(u1)
        netout = C.layers.Dense(output_dim)(d1)        
        return(netout)
    
    def create_sparse_model(self, input_dim, output_dim, hidden_dim, feature_input):
        my_model = C.layers.Sequential([
                C.layers.Embedding(self._num_categories),
                C.layers.Dense(hidden_dim)
                ])
        netout = my_model(feature_input)
        return netout
    
    def get_next_data(self, num_records):
        shape = (1, self.ttyparse.metadata.lines, self.ttyparse.metadata.collumns)
        batch = []
        for _ in range(num_records):
            next_in = np.array(self.ttyparse.get_next_render_flat(), dtype=np.float32)
            next_in = next_in * (1/self._num_categories)
            next_in = next_in.reshape(shape)
            batch.append(next_in)
        np.array(batch, dtype=np.float32)
        return next_in
    

    def moving_average(self, a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

#%%    
if __name__ == '__main__':
    """
    Run from __main__ to allow easier interaction with immediate window

    Hyperparameters
    """
    self = CategoryAutoEncoder()
    input_dim = self.current_input.shape
    output_dim = self.current_input.shape
    hidden_dim = input_dim
    learning_rate = 1e-5
    minibatch_size = 10
    
    """
    Input and output shapes
    """
    feature = C.input((input_dim), np.float32)
    label = C.input((output_dim), np.float32)
#    feature = C.input((input_dim), is_sparse=True)
#    label = C.input((output_dim), np.float32)
    
    #netout = self.create_model(input_dim, output_dim, hidden_dim, feature)
    netout = self.create_model(input_dim, output_dim, hidden_dim, feature)
    loss = C.squared_error(netout, label)    
    evaluation = C.squared_error(netout, label)
    lr_per_minibatch= C.learning_rate_schedule(learning_rate, C.UnitType.minibatch)
    
    learner = C.sgd(netout.parameters, lr=lr_per_minibatch)
    #learner = C.adagrad(netout.parameters, C.learning_rate_schedule(learning_rate, C.UnitType.minibatch))
    
    progress_printer = C.logging.ProgressPrinter(minibatch_size)
    
    trainer = C.Trainer(netout, (loss, evaluation), learner, progress_printer)
    
    plotdata = {"loss":[]}
    for epoch in range(10):
        for i in range(5):
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
    
    x = self.get_next_data(1)[0]
    y = netout.eval({feature: x})
    xy = np.array(np.round(y * 89), dtype=np.int32).reshape((24,80)) +32
    
    for line in xy:
        new_line = ""
        for i in line:
            if i < 32:
                i = ord("ñ")
            new_line += chr(i)            
        print(new_line)
    
#%%
#
#    f = C.input((100), dtype=np.float32)
#    l = C.input((100), dtype=np.float32)
#    exp = C.layers.Sequential([
#             C.layers.Dense(100, C.ops.sigmoid),
#             C.layers.Dense(100)
##            C.layers.Embedding(100),
##            C.layers.Dense(100)            
#            ])
#    m = exp(f)
#    
#    d = np.arange(100, dtype=np.float32)
    
#    e = m.eval({feature: d, label: d})
#    print(e)
    