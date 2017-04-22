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
from cntk.layers import Dense, Sequential, Convolution
import cntk as C
from textmap import Map

np.random.seed(98019)

abs_path = os.path.dirname(os.path.abspath(__file__))

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
    
    
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

#%%
def create_autoencoder(input_dim, output_dim, hidden_dim, feature_input):    
    """
    Create a model with the layers library.
    """
    
    with C.default_options(init = C.glorot_uniform()):
        encode = Dense(input_dim, sigmoid)(feature_input)
        #conv   = Convolution((3,3))(feature_input)
        decode = Dense(output_dim, sigmoid)(encode)
    return(decode)

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
    data_file_path = r'textmap.cntk.txt'   
    """
    Hyperparameters
    """    
    input_dim = 25
    output_dim = 2
    hidden_dim = 25
    smallest_dim = 25
    learning_rate = 0.1
    learning_rate_fine_tuning = 0.0001
    minibatch_size = 20
    
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
    lr_fine_tuning=learning_rate_schedule(learning_rate_fine_tuning, UnitType.minibatch)
    """
    Instantiate the trainer object to drive the model training
    See: https://www.cntk.ai/pythondocs/cntk.learners.html
    """
    learner = sgd(netout.parameters, lr=lr_per_minibatch)  
    learner_fine_tuning = sgd(netout.parameters, lr=lr_fine_tuning)

    # Other learners to try
    #learner = momentum_sgd(netout.parameters, lr=lr_per_minibatch, momentum = momentum_schedule(0.9))
    #learner = adagrad(netout.parameters, lr=lr_per_minibatch) 

    progress_printer = ProgressPrinter(500)
    
    """
    Instantiate the trainer
    See: https://www.cntk.ai/pythondocs/cntk.train.html#module-cntk.train.trainer
    """
    trainer = Trainer(netout, (loss, evaluation), learner, progress_printer)
    trainer_fine_tune = Trainer(netout, (loss, evaluation), learner_fine_tuning, progress_printer)           
#%% 
    """
    Run training
    """
    plotdata = {"loss":[]}
    fine_tuning = False
    for i in range(5000):
        data = training_reader.next_minibatch(minibatch_size, input_map = input_map)
        """
        # This is how to get the Numpy typed data from the reader
        ldata = data[label].asarray()
        fdata = data[feature].asarray()
        """
        lossfine = "NA"
        loss = "NA"
        if fine_tuning:
            trainer_fine_tune.train_minibatch(data)
            loss_fine = trainer_fine_tune.previous_minibatch_loss_average
        else:
            trainer.train_minibatch(data)
            loss = trainer.previous_minibatch_loss_average
            
        
        if fine_tuning == False and loss < 0.25:
            print("Fine tuning!")
            fine_tuning = True

        
        if i % 500 == 0:
            ntldata = data[label].asarray()
            ntfdata = data[feature].asarray()
            network_out = netout.eval({feature: ntfdata[0]})[0]
            print(ntldata[0], network_out)

#            screen_in = ntfdata[0][0]
#            screen_out = netout.eval({feature: ntfdata[0]})[0]
#            m = Map(5,5)
#            m.load_from_data(screen_in)
#            m.display()
#            m.load_from_data(screen_out)
#            m.display()

        
        if not (loss == "NA"):
            plotdata["loss"].append(loss)
        if not (lossfine == "NA"):
            plotdata["loss_fine"].append(loss_fine)
#        if np.abs(trainer.previous_minibatch_loss_average) < 0.0015: #stop once the model is good.
#            break
#%%
    trainer.summarize_training_progress()
    test_data = training_reader.next_minibatch(minibatch_size, input_map = input_map)
    avg_error = trainer.test_minibatch(test_data)
    print("Error rate on an unseen minibatch %f" % avg_error)
    #%%
    ntldata = data[label].asarray()
    ntfdata = data[feature].asarray()
#    for i in range(1):            
#            print("data {},\tevaluation {},\texpected {}".format(
#                    ", ".join(str(v) for v in ntfdata[i][0]),
#                    netout.eval({feature: ntfdata[i]})[0],
#                    ntldata[i][0]))
            
#%%
    import matplotlib.pyplot as plt
    plotdata["avgloss"] = moving_average(plotdata["loss"], 100)
    #plotdata["avgloss"] = plotdata["loss"]
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["avgloss"])
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss')
    plt.show()


    #loss at tail
    plotdata["avgloss"] = moving_average(plotdata["loss_fine"], 1000)
    #plotdata["avgloss"] = plotdata["loss"]
    plt.figure(1)
    plt.subplot(211)
    plt.plot(plotdata["avgloss"])
    plt.xlabel('Minibatch number')
    plt.ylabel('Loss')
    plt.title('Minibatch run vs. Training loss')
    plt.show()

#%%

#    screen_in = ntfdata[0][0]
#    screen_out = netout.eval({feature: ntfdata[0]})[0]
    

#    m = Map(5,5)
#    m.load_from_data(screen_in)
#    m.display()
#    m.load_from_data(screen_out)
#    m.display()
    