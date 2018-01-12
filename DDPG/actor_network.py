# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 13:18:59 2017

@author: dandrews
"""

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, BatchNormalization, Flatten, Conv2D, Input, Dropout
from keras.initializers import RandomUniform
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
from replay_buffer import ReplayBuffer


class ActorNetwork(object):

    def __init__(self, input_shape, output_shape, critic_model, num_rewards):
        # Create actor model
        actor_model = self._create_actor_network(input_shape, output_shape)
        self.reward_num = 0
        self.num_rewards = num_rewards

        # Create actor optimizer that can accept gradients
        # from the critic later
        self.state_input = Input(shape=input_shape)
        out = actor_model(self.state_input)
        self.actor_model = Model(self.state_input,out)
        self.__build_train_fn()
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

        self.critic_model = critic_model
        self.critic_grads = tf.gradients(critic_model.output, critic_model.input)


        # Initialize tensorflow primitives
        self.sess= K.get_session()
        self.sess.run(tf.global_variables_initializer())

        self.actor_model.compile('adam', 'categorical_crossentropy')
        self.actor_target_model.compile('adam', 'categorical_crossentropy')

    def train(self, buffer: tuple, state_input, action_input):
        s_batch, a_batch, r_batch, hra_batch, t_batch, s2_batch = buffer
        q_prediction = self.critic_model.predict([s_batch, a_batch])
        q_val = np.expand_dims(q_prediction[:,self.reward_num], axis=1)
        a_prediction = self.actor_model.predict(s_batch)

        # Turn actor prediction to one-hot
        label = (a_prediction == a_prediction.max(axis=1)[:,None]).astype(np.float32)
        loss = self.train_fn([s_batch,label,q_val])
        return loss

    def __build_train_fn(self):
        """Create a train function
        It replaces `model.fit(X, y)` because we use the output of model and use it for training.
        For example, we need action placeholder
        called `action_one_hot` that stores, which action we took at state `s`.
        Hence, we can update the same action.
        This function will create
        `self.train_fn([state, action_one_hot, discount_reward])`
        which would train the model.
        https://gist.github.com/kkweon/c8d1caabaf7b43317bc8825c226045d2
        """
        action_prob_placeholder = self.actor_model.output
        action_onehot_placeholder = K.placeholder(shape=self.actor_model.output_shape,
                                                  name="action_onehot")
        discount_reward_placeholder = K.placeholder(shape=(None,1),
                                                    name="discount_reward")

        action_prob = K.sum(action_prob_placeholder * action_onehot_placeholder, axis=1)
        log_action_prob = K.log(action_prob)

        loss = - log_action_prob * discount_reward_placeholder
        loss = K.mean(loss)

        adam = Adam()

        updates = adam.get_updates(params=self.actor_model.trainable_weights,
                                   #constraints=[], #constraint?
                                   loss=loss)

        self.train_fn = K.function(inputs=[self.actor_model.input,
                                           action_onehot_placeholder,
                                           discount_reward_placeholder],
                                   outputs=[loss],
                                   updates=updates)


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

    def _create_actor_network(self, input_shape, output_shape):

        actor_model = Sequential(
                [
                Conv2D(filters=5, kernel_size=2, input_shape=input_shape),
                Dense(200,  activation='relu',kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003)),
                #BatchNormalization(),
                #Dropout(0.5),
                Dense(100, activation='relu',kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003)),
                #BatchNormalization(),
#                Dense(output_shape[0],
#                      kernel_initializer=RandomUniform(minval=-0.003, maxval=0.003),
#                      activation='relu'
#                      ),
                Flatten(),
                Dense(output_shape[0], activation='softmax')
                ]
                )

        return actor_model


#%%
if __name__ == '__main__':
    from critic_network import CriticNetwork
    K.clear_session()
    K.set_learning_phase(1)
    input_shape, output_shape = (10,10,3), (4,)
    action_input_shape = output_shape

    cn = CriticNetwork()
    critic_state_input, critic_action_input, critic =\
        cn.create_critic_network(input_shape, output_shape, action_input_shape)
    buffer = ReplayBuffer(10)
    actor_network = ActorNetwork(input_shape, output_shape, critic, 4)

    action = np.array([1,0,0,0])
    s,r,a,s_ =  np.random.rand(10,10,3),\
                np.random.rand(1,output_shape[0]),\
                action,\
                np.random.rand(10,10,3)
    t = False
    for _ in range(10):
        buffer.add(s,a,r,t,s_)
    x = actor_network.train(buffer.sample_batch(10), critic_state_input, critic_action_input)

