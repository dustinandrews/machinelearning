# -*- coding: utf-8 -*-
"""
Created on Fri Jan 12 16:26:51 2018

@author: dandrews

Actor Critic model pair
"""
from keras.models import Sequential, Model
from keras.layers import Conv2D, Flatten, Dense, Multiply, BatchNormalization
from keras.layers import Input, Concatenate
from keras import backend as K
import tensorflow as tf
import numpy as np
K.set_learning_phase(1)

class ActorCritic():
    TAU = 0.1
    _learning_rate = 1e-3 #use change_learning_rate(new_rate)

    def __init__(self, input_shape, action_shape, num_rewards):

        #create models
        shared_base = self._create_shared_based(input_shape)
        critic = self._create_critic_model(shared_base, action_shape)
        actor = self._create_actor_model(shared_base, input_shape, action_shape)
        hybrid = self._create_hybrid_rewards_critic(shared_base, action_shape, num_rewards)
        self.hybrid = hybrid


        self.base = shared_base

        # Setup to get critic gradients
        self.critic = critic
        self.critic_state_input = self.critic.inputs[0]
        self.critic_action_input = self.critic.inputs[1]
        self.critic_grads = tf.gradients(self.critic.output, self.critic_action_input)
        self.sess = K.get_session()
        self.sess.run(tf.global_variables_initializer())

        self.actor = actor

        #K.set_value(self.actor.optimizer.lr, self._learning_rate)
        K.set_value(self.critic.optimizer.lr, self._learning_rate)

    def change_learing_rate(self, learning_rate):
        self._learning_rate = learning_rate
        K.set_value(self.actor.optimizer.lr, self._learning_rate)
        K.set_value(self.critic.optimizer.lr, self._learning_rate)

    def get_learning_rate(self):
        return self._learning_rate

    def _create_shared_based(self, input_shape):
                state = Sequential([
                    ################################
                    # Atari Solving network layout #
                    ################################
                   Conv2D(32, kernel_size=8,
                          strides=4,
                          input_shape=((input_shape)),
                          activation='relu',
                          padding='same',
                          name='Conv2d_1'),
                   BatchNormalization(),
                   Conv2D(64, kernel_size=4, strides=2, activation='relu', name='Conv2d_2', padding='same'),
                   BatchNormalization(),
                   Conv2D(64, kernel_size=3, strides=1, activation='relu', name='Conv2d_3', padding='same'),
                   BatchNormalization(),
                   Dense(128, activation='relu', name='shared_dense_1'),
                   Flatten(),
                   Dense(64, activation='linear', name='shared_output_1' )
                   ])

                return state

    def _create_critic_model(self, shared_state, action_input_shape):
        action =  Sequential([
                Dense(shared_state.layers[-1].output_shape[1], activation='relu',input_shape=action_input_shape, name='action_dense_1'),
                ])

        mult = Multiply()([action.output, shared_state.output])

        merged = Dense(64, activation='relu', name='merged_dense')(mult)
        merged = Dense(32, activation='relu', name='critic_dense')(merged)
        merged = Dense(1, activation='tanh', name='critic_out')(merged)
        model = Model(inputs=[shared_state.input, action.input], outputs=merged)
        model.compile(optimizer='adam', loss='logcosh')
        return model

    def _create_hybrid_rewards_critic(self, shared_state, action_input_shape, num_rewards):
        """
        Create a seperate 'head' for predicting domain knowledge
        """
        action =  Sequential([
                Dense(shared_state.layers[-1].output_shape[1], activation='relu',input_shape=action_input_shape, name='hybrid_dense_1'),
                ])

        mult = Multiply()([action.output, shared_state.output])

        merged = Dense(64, activation='relu', name='hybrid_merged_dense')(mult)
        merged = Dense(32, activation='relu', name='hybrid_dense')(merged)
        merged = Dense(num_rewards, activation='tanh', name='hybrid_out')(merged)
        model = Model(inputs=[shared_state.input, action.input], outputs=merged)
        model.compile(optimizer='adam', loss='logcosh')
        return model



    def _create_actor_model(self, shared_state, input_shape, output_shape):
#        indata = Input(input_shape)
#        shared = shared_state(indata)
        # create actor as new "head" on the critic base
        merged = Dense(50, activation='relu', name='actor_dense_1')(shared_state.output)
        merged = BatchNormalization()(merged)
        merged = Dense(100, activation='relu', name='actor_dense_2')(merged)
        #merged = Dense(50, activation='relu', name='actor_dense_3')(merged)
        merged = Dense(output_shape[0], activation='softmax', name='actor_out')(merged)
        model = Model(shared_state.input,merged)

        self.actor_critic_grad = tf.placeholder(
                tf.float32,
                [None, output_shape[0] ])
        self.actor_grads = tf.gradients(model.output,
                                        model.trainable_weights,
                                        -self.actor_critic_grad
                                        )
        grads = zip(self.actor_grads, model.trainable_weights)
        optimizer = tf.train.AdamOptimizer(self._learning_rate * 10, name="Adam_actor")
        self.optimize = optimizer.apply_gradients(grads)

        model.compile(optimizer=optimizer, loss='categorical_crossentropy')
        return model

    def train_critic(self, s_batch, a_batch, r_batch):
        loss = self.critic.train_on_batch([s_batch, a_batch], r_batch)
        return loss

    def train_hybrid(self, s_batch, a_batch, h_batch):
        loss = self.hybrid.train_on_batch([s_batch, a_batch], h_batch)
        return loss

    def train_actor(self, s_batch, a_batch):

        predictions = self.actor.predict(s_batch)
        grads = self.sess.run(self.critic_grads,
                feed_dict = {
                        self.critic_state_input: s_batch,
                        self.critic_action_input: predictions
                        }
                              )[0]

        self.sess.run(self.optimize,
                feed_dict = {
                    self.actor.input : s_batch,
                    self.actor_critic_grad : grads
                        }
                )
        #TODO: Maybe calulate loss v.s. q_values

        return 0

    def target_train(self, target_actor_critic):
        """
        Nudges target model weights towards this ActorCritic
        """
        self._target_train(self.actor, target_actor_critic.actor)
        # may need to disconect shared layer?
        self._target_train(self.critic, target_actor_critic.critic)


    def _target_train(self, source, target):
        tau = self.TAU
        source_weights = np.array(source.get_weights())
        target_weights = np.array(target.get_weights())
        new_weights = tau * source_weights + (1 - tau) * target_weights
        target.set_weights(new_weights)

    def scale_learning_rate(self, scale=0.1):
        lr = K.get_value(self.critic.optimizer.lr)
        K.set_value(self.critic.optimizer.lr, lr*scale)
        lr = K.get_value(self.actor.optimizer.lr)
        K.set_value(self.actor.optimizer.lr, lr*scale)
        return("New learning rates -  Critic: {}, Actor: {} ".format(
                K.get_value(self.critic.optimizer.lr),
                K.get_value(self.actor.optimizer.lr)
                ))

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


if __name__ == '__main__':
    input_shape = (84,84,3)
    action_shape = (4,)
    actor_critic = ActorCritic(input_shape, action_shape)
    s_batch = np.random.random_sample((10,) + input_shape)
    a_batch = np.random.random_sample((10,) + action_shape)
    r_batch = np.random.random_sample((10,1))
    actor_critic.train_critic(s_batch, a_batch, r_batch)
    actor_critic.train_actor(s_batch, a_batch)
