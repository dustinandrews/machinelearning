# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:56:40 2017

@author: dandrews
"""
import numpy as np
import os
import pickle

class ReplayBuffer(object):

    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.count = 0

        if 'boostrap.pkl' in os.listdir():
            with open('boostrap.pkl', 'rb') as bootstrap:
                self.buffer = pickle.load(bootstrap)
                self.count = sum(1 for elem in self.buffer)

    def add(self, s, a, r, t, s2):
        data = np.array([s, a, r, t, s2, 0.0])
        if self.count == 0:
            self.buffer = data.reshape((1,) + data.shape)
        elif self.count < self.buffer_size:
            self.buffer = np.append(self.buffer,[data], axis=0)
            if self.count % (self.buffer_size // 10) == 0:
                print("Filling buffer {}/{}".format(self.count, self.buffer_size))
        else:
            self.buffer[self.count % self.buffer_size] = data

        self.count += 1

    def size(self):
        return len(self.buffer)

    def sample_batch(self, batch_size: int):
        batch = []
        priority_dist = self.softmax(self.get_sample_weights())

        if self.count < batch_size:
            batch = self.to_batches()
        else:
            indexes = np.random.choice(range(len(self.buffer)), batch_size, p=priority_dist)
            batch = self.get_batches_from_index_list(indexes)
            #batch = random.sample(self.buffer, batch_size)
        return batch


    def to_batches(self):
       return self.get_batches_from_list(self.buffer)

    def get_batches_from_index_list(self, indexes):
        batch = self.buffer[np.array(indexes)]
        return self.get_batches_from_list(batch)

    def get_batches_from_list(self, batch):
        columns = (np.array([r for r in col]) for col in self.buffer.transpose()[:-1])
        return [i for i in columns]

    def get_sample_weights(self):
        weights = self.buffer[:,len(self.buffer[0])-1].astype(np.float32)
        return weights

    def set_sample_weights(self, weights: np.array):
        """ Sets the weights by which the samples are drawn with sample_batch()
        args:
            weights, a one dimensional numpy array the same size as the buffer
            to replace the current weights
        """
        if not type(self.buffer) == type(weights):
            raise ValueError("numpy.ndarray expected. got {}".format(type(weights)))
        shape = weights.shape
        if len(shape) > 1:
            raise ValueError("Must pass a 1 dimensional array, got shape {}".format(shape))

        if len(weights) != len(self.buffer):
            raise ValueError(\
                  "Expected array to match the buffer size {}, got {} elements"\
                  .format(len(self.buffer), len(weights)))
        self.buffer[:,-1] = weights.astype(np.float32)


    def clear(self):
      self.deque.clear()
      self.count = 0

    def softmax(self, a):
        a -= np.min(a)
        a = np.exp(a)
        a /= np.sum(a)
        return a

if __name__ == '__main__':
    rb = ReplayBuffer(10)
    for i in range(11):
        rb.add(np.array([1,1]), 1, i, i % 2 == 0, np.array([1,0]))


    print(rb.size())
    sample = rb.sample_batch(2)
    for s in sample:
        print(s)

