# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================

import numpy as np
import os
from cntk.device import cpu, try_set_default_device
from cntk import Trainer
from cntk.learners import sgd, learning_rate_schedule, UnitType, momentum_sgd, momentum_schedule
from cntk.ops import input, sigmoid, tanh, floor, clip, relu, softplus, log
from cntk.losses import squared_error
from cntk.logging import ProgressPrinter
from cntk.io import CTFDeserializer, MinibatchSource, StreamDef, StreamDefs
from cntk.io import INFINITELY_REPEAT
from cntk.layers import Dense, Sequential
np.random.seed(98019)

#abs_path = os.path.dirname(os.path.abspath(__file__))

def createStandardizedData(path, num_records):
    """
    Create a very simple dataset for testing regression models in the CNTK format
    Models should be able to converge quickly
    x = y1 + 0 + 0
    """
    with open(path, "w") as outfile:    
        for i in range(num_records):
            r = np.random.randint(size=(1, 1), low=0, high=999)[0][0]
            label = (r - 500) / 290
            feature = [label, 0, 0]
            feature = [str(i) for i in feature]
            outfile.write("|labels {} |features {}\n".format(label, " ".join(feature)))


def createStandardizedData2(path, num_records):
    """
    Create a simple equation for testing regression models
    x = (y1-n) * 100 + (y2 -n) * 10 + (y3 - n)
    
    write out raw features and labels as well as a standardized set.
    Shows how models converge quickly with standardized data v.s. raw data.
    """    
    # pre-calculated. Otherwise get them from a sample of the data.
    feature_mean = 50
    feature_std_dist = 10
    label_mean = 500
    label_std_dist = 290
    with open(path, "w") as outfile:    
        for i in range(num_records):
            r = np.random.randint(size=(1, 1), low=0, high=999)[0][0]
            label = r
            standardized_label = standardize(r,label_mean, label_std_dist)
            feature = [ord(c) for c in str(r)]
            while(len(feature) < 3):
                feature.insert(0, 0)
            standardized_feature = [str(i) for i in standardize(feature, feature_mean, feature_std_dist)]
            feature = [str(i) for i in feature]
            outfile.write("|labels {} |features {} |rawlabels {} |rawfeatures {}\n".format(standardized_label, " ".join(standardized_feature), label, " ".join(feature)))


def standardize(a, mean=None, std=None):
    """
    0 center and scale data
    Standardize an np.array to the array mean and standard deviation or specified parameters
    See https://en.wikipedia.org/wiki/Feature_scaling
    """
    if mean == None:
        mean = np.mean(a)
    
    if std == None:
        std = np.std(a)
    a = np.array(a, np.float32)
    n = (a - mean) / std
    return n


def create_reader(path, is_training, input_dim, num_label_classes):
    """
    reads CNTK formatted file with 'labels' and 'features'
    """    
    return MinibatchSource(CTFDeserializer(path, StreamDefs(
        labels = StreamDef(field='labels', shape=num_label_classes),
        features   = StreamDef(field='features', shape=input_dim)
    )), randomize = is_training, max_sweeps = INFINITELY_REPEAT if is_training else 1)   
    
    
def create_reader_raw(path, is_training, input_dim, num_label_classes):
    """
    Reads in the unstardized values.
    """
    return MinibatchSource(CTFDeserializer(path, StreamDefs(
        labels = StreamDef(field='rawlabels', shape=num_label_classes),
        features   = StreamDef(field='rawfeatures', shape=input_dim)
    )), randomize = is_training, max_sweeps = INFINITELY_REPEAT if is_training else 1)        
            
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

#%%
def create_model(input_dim, output_dim, hidden_dim, feature_input):    
    """
    Create a model with the layers library.
    """
    my_model = Sequential ([
            Dense(hidden_dim, tanh),
            Dense(output_dim)
            ])

    netout = my_model(feature_input)   
    return(netout)

#%%

if __name__ == '__main__':
    """
    Run from __main__ to allow easier interaction with immediate window
    """
    
    data_file_path = r'regression_example_data.txt'
    
    #createStandardizedData(data_file_path, num_records =  100000) # a very simple equation
    createStandardizedData2(data_file_path, num_records = 100000) # a slightly complex equation
    
    """
    Hyperparameters
    """    
    input_dim = 3
    output_dim = 1
    hidden_dim = 10
    learning_rate = 0.001
    minibatch_size = 120
    
    """
    Input and output shapes
    """
    feature = input((input_dim), np.float32)
    label = input((output_dim), np.float32)

    """
    Create model, reader and map
    """
    netout = create_model(input_dim, output_dim, hidden_dim, feature)
    training_reader = create_reader(data_file_path, True, input_dim, output_dim)
    input_map = {
    label  : training_reader.streams.labels,
    feature  : training_reader.streams.features
    }
    
    """
    Set loss and evaluation functions
    """
    loss = squared_error(netout, label)    
    evaluation = squared_error(netout, label)
    lr_per_minibatch=learning_rate_schedule(learning_rate, UnitType.minibatch)

    """
    Instantiate the trainer object to drive the model training
    See: https://www.cntk.ai/pythondocs/cntk.learners.html
    """
    learner = sgd(netout.parameters, lr=lr_per_minibatch)    

    # Other learners to try
    #learner = momentum_sgd(netout.parameters, lr=lr_per_minibatch, momentum = momentum_schedule(0.9))
    #learner = adagrad(netout.parameters, lr=lr_per_minibatch) 

    progress_printer = ProgressPrinter(minibatch_size)
    
    """
    Instantiate the trainer
    See: https://www.cntk.ai/pythondocs/cntk.train.html#module-cntk.train.trainer
    """
    trainer = Trainer(netout, (loss, evaluation), learner, progress_printer)
                
#%% 
    """
    Run training
    """
    plotdata = {"loss":[]}
    for i in range(10000):
        data = training_reader.next_minibatch(minibatch_size, input_map = input_map)
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
    test_data = training_reader.next_minibatch(minibatch_size, input_map = input_map)
    avg_error = trainer.test_minibatch(test_data)
    print("Error rate on an unseen minibatch %f" % avg_error)
    #%%
    ntldata = data[label].asarray()
    ntfdata = data[feature].asarray()
    for i in range(10):            
            print("data {},\tevaluation {},\texpected {}".format(
                    ", ".join(str(v) for v in ntfdata[i][0]),
                    netout.eval({feature: ntfdata[i]})[0],
                    ntldata[i][0]))
            
#%%
    import matplotlib.pyplot as plt
    plotdata["avgloss"] = moving_average(plotdata["loss"], 100)
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["avgloss"])
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss')
    plt.show()

    