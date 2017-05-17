# -*- coding: utf-8 -*-
"""
Backprop study

Created on Sun May 14 10:11:32 2017

@author: dandrews
"""

import numpy as np
import cntk as C
from cntk.layers import Tensor
import collections
import matplotlib.pyplot as plt

Hyperparameters = collections.namedtuple(
        'Hyperparamters',
        'hidden_dim learning_rate minibatch_size epochs loss l1reg l2reg stop_loss discount')

class Bandit:
    
    def __init__(self,arms : int):
        self.arms = []
        for i in range(arms):
            self.arms.append(np.random.rand())
        self.action_count = arms
        
    def step(self, action: int):
        reward = 0
        roll = np.random.rand()
        if roll < self.arms[action]:
            reward = 1
        else:
            reward = -1
        return reward
    
class BanditGang:
    
    def __init__(self, bandits : int, arms : int):
        self.bandits = []
        for i in range(bandits):
            self.bandits.append(Bandit(arms))
        self.state = np.zeros(1, dtype=np.float32)
        self.state[0] = np.random.randint(0, bandits)
        self.count = bandits
            
        
    def step(self, action: int):
        reward = self.bandits[int(self.state[0])].step(action)
        self.state[0] = np.random.randint(0, self.count)
        return self.state, reward
    
class Solver:
    
    def __init__(self, num_bandits: int, num_arms: int,  hp: Hyperparameters):         
        self.gang = BanditGang(num_bandits, num_arms)
        self.input_var = C.input(1, dtype=np.float32, name="input_var")
        self.output_var = C.input(num_arms, name="output_var")
        self.label_var = C.input(num_arms, name="label_var")
        self.create_model(hp)
        self.actions = np.arange(num_arms, dtype=np.int32)
        self.softmax = C.softmax(self.output_var)
        self.in_data = np.array((2,), dtype=np.float32) #dummy input for network, for now.
        #self.truth = self.softmax.eval(np.array(self.bandit.arms, dtype=np.float32))
        self.hp = hp
#        self.error = self.get_squared_error()
        self.plotdata = {"loss":[]}
    
    def network(self, input_dim, hidden_dim, output_dim, input_var):
        #self.l1 = C.layers.Dense(hidden_dim, activation=C.sigmoid, name='l_hidden')(input_var)
        #self.l2 = C.layers.Dense(output_dim, name='output_var', activation=C.sigmoid)(self.l1)
        l1 = C.layers.Dense(hidden_dim, activation=C.relu, name='l_hidden1')(input_var)
        l2 = C.layers.Dense(output_dim, name='l_hidden2', activation=C.sigmoid)(l1)               
        return l2
        
    def create_model(self, hp: Hyperparameters):
        self.model = self.network(self.input_var.shape, hp.hidden_dim, self.output_var.shape, self.input_var)
     
    def get_action(self, in_data: np.array):
        output = self.model.eval(in_data)
        prob = self.softmax.eval(output)[0]
        val = np.random.choice(self.actions, p = prob)
        return val
    
        
    """
    Use for regular output/label learning
    """
    def get_next_data(self, size: int, as_policy = True):
        if as_policy:
            return self.collect_policy_data(size)
#        else:                
#            indata = []
#            labeldata = []        
#            for _ in range(size):               
#                indata.append(np.array([np.random.choice(self.actions)], dtype=np.float32))
#                labeldata.append(np.array((self.truth), dtype=np.float32))         
#            indata = np.array(indata, dtype=np.float32)
#            labeldata = np.array(labeldata, dtype=np.float32)
#            return indata, labeldata
    

    """
    Use for Reinforcement learning
    """
    def collect_policy_data(self, size: int):
        return self.collect_policy_data_bandit_gang(size)
    
    def collect_policy_data_bandit_gang(self, size: int):
        print(".", end="")
        state, _ = self.gang.step(0)        
        running_state = np.zeros(size, dtype=np.float32)
        running_reward = np.zeros(size, dtype=np.float32)        
        
        for i in range(size):
            action = self.get_action(state)
            state, reward = self.gang.step(action)
            running_state[i] = state
            running_reward[i] = reward            
                
        current_score = np.average(running_reward)
        credit = np.zeros(shape=(size, self.output_var.shape[0]), dtype=np.float32)
        action_array = np.zeros(self.output_var.shape)
        for i in range(size):
            current_score = current_score * self.hp.discount
            action_array[action_array != 0] = 0
            action_array[running_state[i]] = current_score
            credit[i] = action_array
        
        return running_state, credit
    
    def collect_policy_data_one_bandit(self, size: int):
        print(".", end="")
        data = np.random.choice(self.actions)
        indata = []
        resultdata = [0 for i in range(len(self.bandit.arms))]
        for _ in range(size):
            resultdata[data] += self.bandit.step(data)            
            indata.append([data])
            data = self.get_action(self.in_data)
        resultdata = np.array(resultdata, dtype=np.float32)/size
        credit = []
        for _ in range(size):
            resultdata = resultdata * self.hp.credit
            credit.append(resultdata)
        indata = np.array(indata, dtype=np.float32)
        credit = np.array(credit, dtype=np.float32)         
        return indata, credit
    
        
    def ln_derivative(self, layer):
        w =layer.W.value
        d = sum(w * (1-w))
        return d

    def cross_check_backprop(self):
        prediction = self.model.eval(self.in_data)
        delta = prediction - self.truth
        td = delta * prediction*(1-prediction) 
        l = self.model.find_by_name('l_hidden2')
        cross_check = l.W.value - td * self.hp.learning_rate * delta
        return cross_check

#    def get_actual_error(self):
#        prediction = self.model.eval(self.in_data)
#        delta = prediction - self.truth
#        return delta
        
        
#    def get_squared_error(self):        
#        delta = self.get_actual_error()
#        sumsqerr = .5 * (np.sum(delta*delta))
#        self.error = sumsqerr
#        return self.error

            
    def backprop(self): 
        """
        I had this working for one layer, but I don't quite have it for muliple
        TODO: make this work one day.
        """
        
        pass
    
    def loss_funtion(self):
        loss_fs = {
                'squared_error': C.squared_error(self.model, self.label_var),
                'cross_entropy_with_softmax':   C.cross_entropy_with_softmax(self.model, self.label_var, axis=0),
                'binary_cross_entropy' : C.binary_cross_entropy(self.model, self.label_var),
                'custom_loss' : self.custom_loss(self.model, self.label_var)
                }
        return loss_fs[self.hp.loss]
    
    @C.Function
    def custom_loss(output, target):
        dtype = C.get_data_type(output, target)
        output = C.sanitize_input(output, dtype)
        target = C.sanitize_input(target, dtype)        
        delta = output - target
        pow2 = delta * delta
        loss = C.sqrt(pow2)
        """
        For the moment I don't know how to pull these apart to operate on them with np
        and return something cntk likes.
        
        outvar = C.output_variable(2, np.float32, [])
        x = ????
        return x
        """

        return loss
            

    def train(self, report_freq = 500, as_policy=True):        
        loss = self.loss_funtion()
        evaluation = self.loss_funtion()
        schedule = C.momentum_schedule(self.hp.learning_rate)
        progress_printer = C.logging.ProgressPrinter(num_epochs=self.hp.epochs/self.hp.minibatch_size)
        learner = C.adam(self.model.parameters, 
                     C.learning_rate_schedule(self.hp.learning_rate, C.UnitType.minibatch), 
                     momentum=schedule, 
                     l1_regularization_weight=self.hp.l1reg,
                     l2_regularization_weight=self.hp.l2reg
                     )
        trainer = C.Trainer(self.model, (loss, evaluation), learner, progress_printer)
        self.plotdata = {"loss":[]}
        for epoch in range(self.hp.epochs):             
             indata, label = self.get_next_data(self.hp.minibatch_size, as_policy)
             data = {self.input_var: indata, self.label_var: label}
             trainer.train_minibatch(data)
             loss = trainer.previous_minibatch_loss_average
             if not (loss == "NA"):
                self.plotdata["loss"].append(loss)
             if epoch % report_freq == 0:
                 print()
                 trainer.summarize_training_progress()
             if self.hp.stop_loss > loss:
                 break
        print()
        trainer.summarize_training_progress()
        
    def plot_loss(self):                    
        if len(self.plotdata["loss"]) > 100:
            self.plotdata["avgloss"] = self.moving_average(self.plotdata["loss"], 100)
        else:
            self.plotdata["avgloss"] =self.plotdata["loss"]
        #plotdata["avgloss"] = plotdata["loss"]
        plt.figure(1)    
        plt.subplot(211)
        plt.plot(self.plotdata["avgloss"])
        plt.xlabel('Minibatch number')
        plt.ylabel('Loss')
        #plt.yscale('log', basex=10)
        plt.title('Minibatch run vs. Training loss')
        plt.show()
             
        
    def moving_average(self, a, n=3):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n
    
if __name__ == '__main__':
    
    hp = Hyperparameters(
            hidden_dim=3,
            learning_rate=1e-2,
            minibatch_size=50,
            epochs=10000,
            loss='custom_loss',
            l1reg = 0,
            l2reg = 0,
            stop_loss = 1e-3,
            discount = 0.99
            )
    solver = Solver(3, 3, hp)
    self = solver
    #self.train(as_policy=True)
    #self.plot_loss()
    
    
    
    