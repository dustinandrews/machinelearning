# -*- coding: utf-8 -*-
"""
Created on Sat May 27 08:52:03 2017

@author: Dustin
"""

from matplotlib import pyplot as plt
from textmap import tmap
import numpy as np

import sys
mlpath = r'C:\local\machinelearning'
if not mlpath in sys.path:
    sys.path.append(mlpath)
from daml.parameters import hyperparameters



class RLearn:
    
    def __init__(self, hp: hyperparameters):
        self.env = tmap.Map(5,6)
        self.action_count = env.action_space.n
        

   
    def softmax(x):
        return np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x)))
    
    def update_critic(utility_matrix, observation, new_observation, reward, alpha, gamma):
        u = utility_matrix[observation[0], observation[1]]
        u_t1 = utility_matrix[new_observation[0], new_observation[1]]
        delta = reward + gamma * u_t1 - u
        utility_matrix[observation[0], observation[1]] += alpha * delta
        return utility_matrix, delta
    
    def update_actor(state_action_matrix, observation, action, delta, beta_matrix=None):
        col = observation1[1] + (observation[0] * 4)
        if beta_matrix is None:
            beta = 1
        else: 
            beta = 1 / beta_matrix[action, col]
        state_action_matrix[action, col] += beta * delta
        return state_action_matrix
    
    def train_model(self):
        for epoch in range(self.hp.epochs):
            for step in range(1000):
                col = observation[1] + (observation[0] * 4)
                action_array = state_action_matrix[:, col]
                action_distrubution = softmax(action_array)
                action = np.random.choice(self.action_count, 1, p=action_distrubution)
                new_observation, reward, done, data = env.step(action)
                utility_matrix, delta = self.update_critic(
                        state_action_matrix,
                        observation,
                        reward,
                        alpha,
                        gamma
                        )
                state_action_matrix = self.update_actor(
                        state_action_matrix,
                        observation,
                        action,
                        delta,
                        beta_matrix=None
                        )
                observation = new_observation
                if done:
                    break
    
    
    
#%%
if __name__ == "__main__":
     
        hp = hyperparameters(
             hidden_dim=500, 
             learning_rate=1e-3, 
             minibatch_size=100,
             epochs=100,
             optimizer='rmsprop',
             loss=mse
             )
    
    
    rl = RLearn(hp)