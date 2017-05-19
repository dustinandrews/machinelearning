# -*- coding: utf-8 -*-
"""
Backprop study

Created on Sun May 14 10:11:32 2017

@author: dandrews
"""

import numpy as np
import cntk as C
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
            reward = 0
        return reward
    
class BanditGang:
    
    def __init__(self, bandits : int, arms : int):
        self.bandits = []
        for i in range(bandits):
            self.bandits.append(Bandit(arms))
        self.state = np.zeros(1, dtype=np.float32)
        self.state[0] = np.random.randint(0, bandits)
        self.count = bandits
        self.arms = arms
        self.truth = self.get_truth()
            
        
    def step(self, action: int):
        reward = self.bandits[int(self.state[0])].step(action)
        self.state[0] = np.random.randint(0, self.count)
        return self.state, reward
    
    def get_truth(self):
        result = np.zeros(shape=(self.count, self.arms), dtype=np.float32)
        for i in range(self.count):
            result[i] = self.bandits[i].arms
            
        return result
            
    
    
class Solver:
    
    def __init__(self, num_bandits: int, num_arms: int,  hp: Hyperparameters):         
        self.gang = BanditGang(num_bandits, num_arms)
        self.input_var = C.input(2, dtype=np.float32, name="input_var") #state and proposed action
        self.output_var = C.input(1, name="output_var")
        self.label_var = C.input(1, name="label_var")
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
     
    def get_action(self, in_data=None):
#        in_data = np.array(in_data, dtype=np.float32)
#        output = self.model.eval(in_data)
#        prob = self.softmax.eval(output)[0]
        val = np.random.choice(self.actions)
        return val
    
        
    """
    Use for regular output/label learning
    """
    def get_next_data(self, size: int, as_policy = True):
        
        
        indata = np.zeros(shape=(size, self.input_var.shape[0]), dtype=np.float32)
        labeldata = np.zeros(shape=(size, self.output_var.shape[0]), dtype=np.float32)
        total = 0
        for i in range(size):
            action = np.random.choice(self.actions)
            indata[i] = [action, self.gang.state]
            _, r = self.gang.step(action)
            labeldata[i] = r
            total += r
        return indata, labeldata, total
            
        
        
#        if as_policy:
#            return self.collect_policy_data(size)
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
        running_state = np.zeros(shape=(size, 1), dtype=np.float32)
        running_reward = np.zeros(size, dtype=np.float32)        
        
        for i in range(size):
            action = self.get_action(state)
            state, reward = self.gang.step(action)
            running_state[i] = [state]
            running_reward[i] = reward
                
        current_score = np.average(running_reward) * 100        
        credit = np.zeros(shape=(size, 1), dtype=np.float32)
        action_array = np.zeros(1)
        for i in range(size):
            current_score = current_score * self.hp.discount            
            action_array[0] = current_score
            credit[i] = action_array        
        return running_state, credit, np.sum(running_reward)
    
        
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
            

    def train(self, report_freq = 500, as_policy=True):        
        #loss = C.ops.minus(0, C.ops.argmin(self.model) -  C.ops.argmin(self.model) + C.ops.minus(self.label_var, 0))
        loss = C.squared_error(self.model, self.label_var)
        evaluation = C.squared_error(self.model, self.label_var)
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
             indata, label, total_reward = self.get_next_data(self.hp.minibatch_size, as_policy)
             data = {self.input_var: indata, self.label_var: label}
             trainer.train_minibatch(data)
             loss = trainer.previous_minibatch_loss_average
             if not (loss == "NA"):
                self.plotdata["loss"].append(loss)
             if epoch % report_freq == 0:
                 print()
                 print("last epoch total reward: {}".format(total_reward))
                 trainer.summarize_training_progress()
                 print()
#             if self.hp.stop_loss > loss:
#                 break
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
            hidden_dim=10,
            learning_rate=1e-2,
            minibatch_size=150,
            epochs=100000,
            loss='custom_loss',
            l1reg = 0,
            l2reg = 0,
            stop_loss = 1e-3,
            discount = 0.99
            )
    solver = Solver(3, 3, hp)
    self = solver
    
    ind = np.array([0,0], dtype=np.float32)    
    for i in range(3):
        for j in range(3):
            ind[0] = i
            ind[1] = j
            e = self.model.eval(ind)
            print(i, j, e, self.gang.truth[i][j])    

    
    self.train(as_policy=True)
    self.plot_loss()
    
   
    ind = np.array([0,0], dtype=np.float32)    
    for i in range(3):
        for j in range(3):
            ind[0] = i
            ind[1] = j
            e = self.model.eval(ind)
            print(i, j, e, self.gang.truth[i][j])    
    