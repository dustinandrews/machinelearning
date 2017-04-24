# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 21:05:05 2017

@author: Dustin

AI learner to play games.
"""
from __future__ import print_function
from __future__ import division

from IPython.display import Image

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
import seaborn as sns

style.use('ggplot')

import gym

isFast = False
import random, numpy, math, os

#from keras.models import Sequential
#from keras.layers import *
#from keras.optimizers import *
from cntk import *
from cntk.layers import *
from cntk.ops.sequence import input
# Select the right target device when this notebook is being tested:
if 'TEST_DEVICE' in os.environ:
    if os.environ['TEST_DEVICE'] == 'cpu':
        device.try_set_default_device(device.cpu())
    else:
        device.try_set_default_device(device.gpu(0))
#env = gym.make('CartPole-v0')
from textmap import Map
env = Map(5,5)

STATE_COUNT  = env.observation_space.n
ACTION_COUNT = env.action_space.n

STATE_COUNT, ACTION_COUNT
# Targetted reward
REWARD_TARGET = 100 if isFast else 120
# Averaged over these these many episodes
BATCH_SIZE_BASELINE = 20 if isFast else 50

H = 64 # hidden layer size

class Brain:
    def __init__(self):
        self.params = {}
        self.model, self.trainer, self.loss = self._create()
        # self.model.load_weights("cartpole-basic.h5")

    def _create(self):
        observation = input(STATE_COUNT, np.float32, name="s")
        q_target = input(ACTION_COUNT, np.float32, name="q")

        # model = Sequential()
        # model.add(Dense(output_dim=64, activation='relu', input_dim=STATE_COUNT))
        # model.add(Dense(output_dim=ACTION_COUNT, activation='linear'))

        # Following a style similar to Keras
        l1 = Dense(H, activation=relu)
        l2 = Dense(ACTION_COUNT)
        unbound_model = Sequential([l1, l2])
        model = unbound_model(observation)

        self.params = dict(W1=l1.W, b1=l1.b, W2=l2.W, b2=l2.b)

        lr = 0.00025
        # opt = RMSprop(lr=0.00025)
        # model.compile(loss='mse', optimizer=opt)

        # loss='mse'
        loss = reduce_mean(square(model - q_target), axis=0)
        meas = reduce_mean(square(model - q_target), axis=0)

        # optimizer=opt
        lr_schedule = learning_rate_schedule(lr, UnitType.minibatch)
        learner = sgd(model.parameters, lr_schedule, gradient_clipping_threshold_per_sample=10)
        trainer = Trainer(model, (loss, meas), learner)

        # CNTK: return trainer and loss as well
        return model, trainer, loss

    def train(self, x, y, epoch=1, verbose=0):
        #self.model.fit(x, y, batch_size=64, nb_epoch=epoch, verbose=verbose)
        arguments = dict(zip(self.loss.arguments, [x,y]))
        updated, results =self.trainer.train_minibatch(arguments, outputs=[self.loss.output])

    def predict(self, s):
        return self.model.eval([s])
class Memory:   # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)
MEMORY_CAPACITY = 100000
BATCH_SIZE = 64

GAMMA = 0.99 # discount factor

MAX_EPSILON = 1
MIN_EPSILON = 0.01 # stay a bit curious even when getting old
LAMBDA = 0.0001    # speed of decay

class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self):
        self.brain = Brain()
        self.memory = Memory(MEMORY_CAPACITY)

    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, ACTION_COUNT-1)
        else:
            return numpy.argmax(self.brain.predict(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self):
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = numpy.zeros(STATE_COUNT)


        # CNTK: explicitly setting to float32
        states = numpy.array([ o[0] for o in batch ], dtype=np.float32)
        states_ = numpy.array([(no_state if o[3] is None else o[3]) for o in batch ], dtype=np.float32)

        p = agent.brain.predict(states)
        p_ = agent.brain.predict(states_)

        # CNTK: explicitly setting to float32
        x = numpy.zeros((batchLen, STATE_COUNT)).astype(np.float32)
        y = numpy.zeros((batchLen, ACTION_COUNT)).astype(np.float32)

        for i in range(batchLen):
            s, a, r, s_ = batch[i]

            # CNTK: [0] because of sequence dimension
            t = p[0][i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * numpy.amax(p_[0][i])

            x[i] = s
            y[i] = t

        self.brain.train(x, y)
        
def plot_weights(weights, figsize=(7,5)):
    '''Heat map of weights to see which neurons play which role'''
    sns.set(style="white")
    f, ax = plt.subplots(len(weights), figsize=figsize)
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    
    for i, data in enumerate(weights):
        axi = ax if len(weights)==1 else ax[i]
        if isinstance(data, tuple):
            w, title = data
            axi.set_title(title)
        else:
            w = data
                
        sns.heatmap(w.asarray(), cmap=cmap, square=True, center=True, #annot=True,
                    linewidths=.5, cbar_kws={"shrink": .25}, ax=axi)
def epsilon(steps):
    return MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * np.exp(-LAMBDA * steps)
    plt.plot(range(10000), [epsilon(x) for x in range(10000)], 'r')
    plt.xlabel('step');plt.ylabel('$\epsilon$')

TOTAL_EPISODES = 200 if isFast else 3000

def run(agent):
    s = env.reset()
    R = 0

    while True:
        # Uncomment the line below to visualize the cartpole
#        if R % 50 == 0:
#            env.render()

        # CNTK: explicitly setting to float32
        a = agent.act(s.astype(np.float32))

        s_, r, done, info = env.step(a)

        if done: # terminal state
            s_ = None

        agent.observe((s, a, r, s_))
        agent.replay()

        s = s_
        R += r

        if done:
            env.render()
            return R


def dqn():
    global agent
    agent = Agent()
    
    episode_number = 0
    reward_sum = 0
    while episode_number < TOTAL_EPISODES:
        print(episode_number)
        reward_sum += run(agent)
        episode_number += 1
        if episode_number % BATCH_SIZE_BASELINE == 0:
            print('Episode: %d, Average reward for episode %f.' % (episode_number,
                                                                   reward_sum / BATCH_SIZE_BASELINE))
            if episode_number%50==0:
                plot_weights([(agent.brain.params['W1'], 'Episode %i $W_1$'%episode_number)], figsize=(14,5))
            if reward_sum / BATCH_SIZE_BASELINE > REWARD_TARGET:
                print('Task solved in %d episodes' % episode_number)
                plot_weights([(agent.brain.params['W1'], 'Episode %i $W_1$'%episode_number)], figsize=(14,5))
                break
            reward_sum = 0
    agent.brain.model.save('dqn.mod')


def run_dqn_from_model():
    import cntk as C
    #env = gym.make('CartPole-v0')
    
    num_episodes = 10  # number of episodes to run
    
    modelPath = 'dqn.mod'
    root = C.load_model(modelPath)
    
    for i_episode in range(num_episodes):
        print(i_episode)
        observation = env.reset()  # reset environment for new episode
        done = False
        while not done: 
            if not 'TEST_DEVICE' in os.environ:
                env.render()
            action = np.argmax(root.eval([observation.astype(np.float32)]))
            observation, reward, done, info  = env.step(action)
            

"""
Policy Gradient
"""
          
def discount_rewards(r, gamma=0.999):
    """Take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def plot_discounts():
    discounted_epr = discount_rewards(np.ones(10))
    f, ax = plt.subplots(1, figsize=(5,2))
    sns.barplot(list(range(10)), discounted_epr, color="steelblue")
    discounted_epr_cent = discounted_epr - np.mean(discounted_epr)
    discounted_epr_norm = discounted_epr_cent/np.std(discounted_epr_cent)
    f, ax = plt.subplots(1, figsize=(5,2))
    sns.barplot(list(range(10)), discounted_epr_norm, color="steelblue")
    discounted_epr = discount_rewards(np.ones(10), gamma=0.5)
    discounted_epr_cent = discounted_epr - np.mean(discounted_epr)
    discounted_epr_norm = discounted_epr_cent/np.std(discounted_epr_cent)
    f, ax = plt.subplots(2, figsize=(5,3))
    sns.barplot(list(range(10)), discounted_epr, color="steelblue", ax=ax[0])
    sns.barplot(list(range(10)), discounted_epr_norm, color="steelblue", ax=ax[1])

def policy_gradient():
    import cntk as C
    global TOTAL_EPISODES
    TOTAL_EPISODES = 2000 if isFast else 10000
    
    
    D = 4  # input dimensionality
    H = 10 # number of hidden layer neurons
    
    observations = input(STATE_COUNT, np.float32, name="obs")
    
    W1 = C.parameter(shape=(STATE_COUNT, H), init=C.glorot_uniform(), name="W1")
    b1 = C.parameter(shape=H, name="b1")
    layer1 = C.relu(C.times(observations, W1) + b1)
    
    W2 = C.parameter(shape=(H, ACTION_COUNT), init=C.glorot_uniform(), name="W2")
    b2 = C.parameter(shape=ACTION_COUNT, name="b2")
    score = C.times(layer1, W2) + b2
    # Until here it was similar to DQN
    
    probability = C.sigmoid(score, name="prob")
    input_y = input(1, np.float32, name="input_y")
    advantages = input(1, np.float32, name="advt")
    
    loss = -C.reduce_mean(C.log(C.square(input_y - probability) + 1e-4) * advantages, axis=0, name='loss')
    
    lr = 0.001
    lr_schedule = learning_rate_schedule(lr, UnitType.sample)
    sgd = C.sgd([W1, W2], lr_schedule)
    
    gradBuffer = dict((var.name, np.zeros(shape=var.shape)) for var in loss.parameters if var.name in ['W1', 'W2', 'b1', 'b2'])
    
    xs, hs, label, drs = [], [], [], []
    running_reward = None
    reward_sum = 0
    episode_number = 1
    
    observation = env.reset()
    
    while episode_number <= TOTAL_EPISODES:
        x = np.reshape(observation, [1, STATE_COUNT]).astype(np.float32)
    
        # Run the policy network and get an action to take.
        prob = probability.eval(arguments={observations: x})[0][0][0]
        action = 1 if np.random.uniform() < prob else 0
    
        xs.append(x)  # observation
        # grad that encourages the action that was taken to be taken
    
        y = 1 if action == 0 else 0  # a "fake label"
        label.append(y)
    
        # step the environment and get new measurements
        observation, reward, done, info = env.step(action)
        reward_sum += float(reward)
    
        # Record reward (has to be done after we call step() to get reward for previous action)
        drs.append(float(reward))
    
        if done:
            # Stack together all inputs, hidden states, action gradients, and rewards for this episode
            epx = np.vstack(xs)
            epl = np.vstack(label).astype(np.float32)
            epr = np.vstack(drs).astype(np.float32)
            xs, label, drs = [], [], []  # reset array memory
    
            # Compute the discounted reward backwards through time.
            discounted_epr = discount_rewards(epr)
            # Size the rewards to be unit normal (helps control the gradient estimator variance)
            discounted_epr -= np.mean(discounted_epr)
            discounted_epr /= np.std(discounted_epr)
    
            # Forward pass
            arguments = {observations: epx, input_y: epl, advantages: discounted_epr}
            state, outputs_map = loss.forward(arguments, outputs=loss.outputs,
                                              keep_for_backward=loss.outputs)
    
            # Backward psas
            root_gradients = {v: np.ones_like(o) for v, o in outputs_map.items()}
            vargrads_map = loss.backward(state, root_gradients, variables=set([W1, W2]))
    
            for var, grad in vargrads_map.items():
                gradBuffer[var.name] += grad
    
            # Wait for some batches to finish to reduce noise
            if episode_number % BATCH_SIZE_BASELINE == 0:
                grads = {W1: gradBuffer['W1'].astype(np.float32),
                         W2: gradBuffer['W2'].astype(np.float32)}
                updated = sgd.update(grads, BATCH_SIZE_BASELINE)
    
                # reset the gradBuffer
                gradBuffer = dict((var.name, np.zeros(shape=var.shape))
                                  for var in loss.parameters if var.name in ['W1', 'W2', 'b1', 'b2'])
    
                print('Episode: %d. Average reward for episode %f.' % (episode_number, reward_sum / BATCH_SIZE_BASELINE))
    
                if reward_sum / BATCH_SIZE_BASELINE > REWARD_TARGET:
                    print('Task solved in: %d ' % episode_number)
                    break
    
                reward_sum = 0    
            observation = env.reset()  # reset env
            episode_number += 1    
    probability.save('pg.mod')

def create_dqn_without_lib():
    observation = input(STATE_COUNT, np.float32, name="s")
    W1 = parameter(shape=(STATE_COUNT, H), init=glorot_uniform(), name="W1")
    b1 = parameter(shape=H, name="b1")
    layer1 = relu(times(observation, W1) + b1)
    W2 = parameter(shape=(H, ACTION_COUNT), init=glorot_uniform(), name="W2")
    b2 = parameter(shape=ACTION_COUNT, name="b2")
    model = times(layer1, W2) + b2
    W1.shape, b1.shape, W2.shape, b2.shape, model.shape


dqn()