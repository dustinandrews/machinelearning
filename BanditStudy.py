# -*- coding: utf-8 -*-
"""
Backprop study

Created on Sun May 14 10:11:32 2017

@author: dandrews
"""

import numpy as np
import cntk as C
import collections

Hyperparameters = collections.namedtuple('Hyperparamters', 'hidden_dim learning_rate minibatch_size epochs')

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
    
class Solver:
    
    def __init__(self, num_arms: int,  hp: Hyperparameters):        
        self.bandit = Bandit(num_arms)
        self.input_var = C.input(1, dtype=np.float32)
        self.output_var = C.input(num_arms)
        self.create_model(hp)
        self.actions = np.arange(num_arms, dtype=np.int32)
        self.softmax = C.softmax(self.output_var)
        self.in_data = np.array((1,), dtype=np.float32) #dummy input for network, for now.
        self.truth = self.softmax.eval(np.array(self.bandit.arms, dtype=np.float32))
        self.hp = hp
        self.error = self.get_error()
    
    def network(self, input_dim, hidden_dim, output_dim, input_var):
        
        #self.l1 = C.layers.Dense(hidden_dim, activation=C.sigmoid, name='l_hidden')(input_var)
        #self.l2 = C.layers.Dense(output_dim, name='l_output', activation=C.sigmoid)(self.l1)
        l1 = C.layers.Dense(hidden_dim, activation=C.sigmoid, name='l_hidden')(input_var)
        l2 = C.layers.Dense(output_dim, name='l_output', activation=C.sigmoid)(l1)        
        self.reverse_layers = ['l_output', 'l_hidden']        
        return l2
        
    def create_model(self, hp: Hyperparameters):
        self.model = self.network(self.input_var.shape, hp.hidden_dim, self.output_var.shape, self.input_var)
     
    def get_action(self, in_data: np.array):
        output = self.model.eval(in_data)
        prob = self.softmax.eval(output)[0]
        val = np.random.choice(self.actions, p = prob)
        return val
    
    def step(self):
        action = self.get_action(self.in_data)
        reward = self.bandit.step(action)
        print(reward)
        
    def ln_derivative(self, layer):
        w =layer.W.value
        d = w * (1-w)
        return d
#        prediction = self.model.eval(self.in_data)
#        delta = prediction - self.truth
#        td = delta * prediction*(1-prediction)
#        self.error = np.sum(np.sqrt(delta*delta))
#        return td

    def get_error(self):
        prediction = self.model.eval(self.in_data)
        delta = prediction - self.truth
        self.error = np.sum(np.sqrt(delta*delta))
        return self.error

            
    def backprop(self):
        """
        Not right
        """
        w = {}        
        error = self.get_error()
        # get derivatives and cache new W
        for lname in self.reverse_layers:
            l = self.model.find_by_name(lname)
            d = self.ln_derivative(l)
            new_w = l.W.value - d * self.hp.learning_rate * error # probably here
            w[lname]= new_w
        
        # apply new w
        for lname in self.reverse_layers:
            l = self.model.find_by_name(lname)
            l.W.value = w[lname]
    

    def train(self, iterations: int):
        print("error:" + str(self.get_error()) )
        for i in range(iterations):
            self.backprop()
            if i % 100 == 0:
                print("error:" + str(self.get_error()) )
        print("error:" + str(self.get_error()) )

        
    
if __name__ == '__main__':
    
    hp = Hyperparameters(hidden_dim=3, learning_rate=1e-1, minibatch_size=20, epochs=100)
    solver = Solver(3, hp)
    self = solver
    print(solver.error)
    
    