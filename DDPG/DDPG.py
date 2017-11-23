# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:05:34 2017

@author: dandrews
"""
import sys
sys.path.append('D:/local/machinelearning/textmap')
from tmap import Map

from replay_buffer import ReplayBuffer
from actor_network import ActorNetwork
from critic_network import CriticNetwork
import keras
import numpy as np

class DDPG(object):
    buffer_size = 100
    input_shape = (10,10)
    
    def __init__(self):
        e = Map(self.input_shape[0],self.input_shape[1])        
        self.output_shape = e.action_space.n
        self.action_input_shape = (e.action_space.n,)
        
        self.environment = e
        
        self.buffer = ReplayBuffer(self.buffer_size)
        
        actor_network = ActorNetwork()        
        self.actor = actor_network.create_actor_network(
                self.input_shape,
                self.output_shape)
        self.actor_target = actor_network.create_actor_network(
                self.input_shape,
                self.output_shape)
        
        critic_network = CriticNetwork()
        self.critic = critic_network.create_critic_network(
                self.input_shape,
                self.output_shape,
                self.action_input_shape
                )
        
        self.critic_target = critic_network.create_critic_network(
                self.input_shape,
                self.output_shape,
                self.action_input_shape
                )
        
        

    def step(self):
        state = np.expand_dims(self.environment.data_normalized(), axis=0)        
        prediction = self.actor.predict([state],1)
        return prediction
        
    def target_train(self, source: keras.models.Model, target: keras.models.Model):
        source_weights = source.get_weights()
        target_weights = self.target.get_weights()
        for i in range(len(source_weights)):
            target_weights[i] = self.TAU * source_weights[i] +\
            (1. - self.TAU) * target_weights[i]
        self.actor_target
        
    
            


if __name__ == '__main__':
    ddpg = DDPG()
    m = Map(10,10)
    m.render()