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
import random
from keras import backend as K
K.clear_session()

class DDPG(object):
    buffer_size = 100
    input_shape = (10,10)
    decay = 0.9
    
    
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
    
    def fill_replay_buffer(self, from_actor=True):
        e = self.environment
        for i in range(self.buffer_size):
            if e.done:
                e.reset()            
            a = self.get_action(from_actor)
            s = e.data_normalized()
            (s_, r, t, info) = e.step(a)            
            self.buffer.add(s, a, r, t, s_)
            print(self.buffer.count)
                
    def train_critic_from_buffer(self):        
        for i in range(100):
           s_batch, a_batch, r_batch, t_batch, s2_batch = self.buffer.sample_batch(10)
           self.critic.train_on_batch([s_batch, a_batch], r_batch)
           
            
          

    def get_action(self, from_actor):
        if from_actor:
            pass
        else:            
            action = random.choice(list(self.environment._actions.keys()))
        return action


if __name__ == '__main__':
    ddpg = DDPG()
    m = Map(10,10)
    m.render()
    pred = ddpg.step()
    print(pred) 
    ddpg.fill_replay_buffer(from_actor=False)
    s_batch, a_batch, r_batch, t_batch, s2_batch = ddpg.buffer.sample_batch(10)
