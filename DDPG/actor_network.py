# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:18:59 2017

@author: dandrews
"""

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Flatten, Conv2D, Input
from keras.initializers import RandomUniform
from keras import backend as K
import numpy as np

class ActorNetwork(object):

    def __init__(self, input_shape, output_shape, critic_model):
        # Create actor model
        actor_model = self._create_actor_network(input_shape, output_shape)
        
        # Create actor optimizer that can accept gradients 
        # from the critic later
        self.state_input = Input(shape=input_shape)
        out = actor_model(self.state_input)        
        self.actor_model = Model(self.state_input,out)
        self.actor_input = self.state_input
        
        self.actor_critic_grad = tf.placeholder(tf.float32, 
            [None, output_shape[0]]) 
        
        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output, 
            actor_model_weights, -self.actor_critic_grad)
        grads = zip(self.actor_grads, actor_model_weights)
        
        # create the optimizer
        self._optimize =  tf.train.AdamOptimizer().apply_gradients(grads)        
        
        # Create the actor target model
        actor_target = self._create_actor_network(input_shape, output_shape)
        target_out = actor_target(self.state_input)
        self.actor_target_model = Model(self.state_input, target_out)
        
        self.critic_grads = tf.gradients(critic_model.output, critic_model.input)
        
        # Initialize tensorflow primitives
        self.sess= K.get_session()
        self.sess.run(tf.global_variables_initializer())
        
        self.actor_model.compile('adam', 'mse')
        self.actor_target_model.compile('adam', 'mse')
    
    def train(self, buffer, state_input, action_input):
        for item in buffer:
            s, a, r, t, s_ = item
            s = np.array([s])
            prediction = self.actor_model.predict(s)
            
            grads = self.sess.run(
                    self.critic_grads,
                    feed_dict = {
                            state_input: s,
                            action_input: prediction
                                }
                                  )[1]
            
            x = self.sess.run(self._optimize,
                          feed_dict =
                          {
                              self.state_input: s,
                              self.actor_critic_grad: grads
                          }
                          )
        return x
    
    def _create_actor_network(self, input_shape, output_shape):
        
        actor_model = Sequential(
                [
                Conv2D(filters=5, kernel_size=1, input_shape=input_shape),
               #Flatten(input_shape=input_shape),
                Dense(100,  activation='relu',kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003)),                
                BatchNormalization(),
                Dense(100, activation='relu',kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003)),
                BatchNormalization(),
                Dense(output_shape[0], 
                      kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003),
                      activation='relu'
                      ),
                Flatten(),
                Dense(output_shape[0], activation='softmax')
                ]                     
                )                

        return actor_model
    
    
    
if __name__ == '__main__':
    from critic_network import CriticNetwork
    K.clear_session()
    K.set_learning_phase(1)
    input_shape, output_shape = (10,10,3), (4,)
    action_input_shape = output_shape    
    
    cn = CriticNetwork()
    critic_state_input, critic_action_input, critic =\
        cn.create_critic_network(input_shape, output_shape, action_input_shape)
    
    actor_network = ActorNetwork(input_shape, output_shape, critic)
    
    s,r,a,s_ = np.random.rand(1,10,10,3), 1, np.random.rand(1,output_shape[0]), np.random.rand(10,10,3)
    buffer = [[s,r,a,s_], [s,r,a,s_]]
    x = actor_network.train(buffer, critic_state_input, critic_action_input)
    
