# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:18:59 2017

@author: dandrews
"""

import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input
from keras import backend as K
import numpy as np
from replay_buffer import ReplayBuffer
from critic_network import CriticNetwork

class ActorNetwork(object):
    optimizer = 'adam'
    loss = 'categorical_crossentropy'


    def __init__(self, input_shape, output_shape, critic_network: CriticNetwork):
        if not isinstance(critic_network,CriticNetwork):
            raise ValueError("critic_network must be instance of CriticNetwork")
        # Create actor model
        actor_model = self._create_actor_network(input_shape, output_shape[0], critic_network)

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
        actor_target = self._create_actor_network(input_shape, output_shape[0], critic_network)
        target_out = actor_target(self.state_input)
        self.actor_target_model = Model(self.state_input, target_out)

        self.critic_target_model = critic_network.critic_target_model

        # Initialize tensorflow primitives
        self.sess= K.get_session()
        self.sess.run(tf.global_variables_initializer())

        self.actor_model.compile('adam', 'categorical_crossentropy')
        self.actor_target_model.compile('adam', 'categorical_crossentropy')

    def train(self, buffer: tuple, state_input, action_input):
        s_batch, a_batch, r_batch, t_batch, s2_batch = buffer
        #q_val = r_batch
        q_val = self.critic_target_model.predict([s_batch,a_batch])
        #q_norm = self.scaler.fit_transform(q_val) #Normalize to 0-1
        #q_norm = (q_val - q_val.min()) / (q_val.max() - q_val.min())

        a_prediction = self.actor_model.predict(s_batch)
        # Turn actor prediction to one-hot
        label = (a_prediction == a_prediction.max(axis=1)[:,None]).astype(np.float32) * q_val
        loss = -self.actor_model.train_on_batch(s_batch, label)
        return loss

    def cross_entropy(self, X ,y):
        """
        X is the output from fully connected layer (num_examples x num_classes)
        y is labels (num_examples x 1)
        """
        m = y.shape[0]
        #p = softmax(X) # should already be softmax
        p = X
        log_likelihood = -np.log(p[range(m)],y)
        loss = np.sum(log_likelihood) / m
        return loss

    def _create_actor_network(self, input_shape, output_shape, critic_model: CriticNetwork):
        indata = Input(input_shape)
        shared = critic_model.shared_state_network(indata)
        # create actor as new "head" on the critic base
        merged = Dense(100, activation='relu', name='actor_dense_1')(shared)
        merged = Dense(50, activation='relu', name='actor_dense_2')(merged)
        merged = Dense(output_shape, activation='softmax', name='actor_out')(merged)
        model = Model(indata,merged)
        model.compile(optimizer=self.optimizer, loss=self.loss)
        return model


##%%
if __name__ == '__main__':
    from critic_network import CriticNetwork
    K.clear_session()
    K.set_learning_phase(1)
    input_shape, output_shape = (2,2,1), (4,)
    action_input_shape = output_shape

    cn = CriticNetwork()
    cn.create_critic_network(input_shape, output_shape, action_input_shape)
    critic_state_input, critic_action_input = cn.state_input, cn.action_input

    buffer = ReplayBuffer(10)
    actor_network = ActorNetwork(input_shape, output_shape, cn)

    action = np.array([1,0,0,0])
    s,r,a,s_ =  np.random.rand(10,10,3),\
                np.random.rand(1,output_shape[0]),\
                action,\
                np.random.rand(10,10,3)
    t = False
    for _ in range(10):
        buffer.add(s,a,r,t,s_)
    x = actor_network.train(buffer.sample_batch(10), critic_state_input, critic_action_input)

