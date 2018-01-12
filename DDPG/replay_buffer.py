# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:56:40 2017

@author: dandrews
"""
import numpy as np
import os
import pickle
from collections import namedtuple

Record = namedtuple('move', ['s','a','r', 'hra','t','s_'])


class ReplayBuffer(object):

    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.count = 0
        r = Record('s','a','r', 'hra','t','s_')
        self._record_len = len(r)

        if 'boostrap.pkl' in os.listdir():
            with open('boostrap.pkl', 'rb') as bootstrap:
                self.buffer = pickle.load(bootstrap)
                self.count = sum(1 for elem in self.buffer)

    def add(self, data):
        if self.count == 0:
            self.buffer = np.array([data])
            self.sample_weights = np.array([0])
        elif self.count < self.buffer_size:
            self.buffer = np.append(self.buffer,[data], axis=0)
            self.sample_weights = np.append(self.sample_weights, 0)
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
        records = self.buffer[np.array(indexes)]
        batch = self.get_batches_from_list(records)
        return batch

    """
    Batch is one array for each collumn in the records
    to make things Keras ready
    """
    def get_batches_from_list(self, records):
        l = self._record_len
        cols = [[] for i in range(l)]
        for record in records:
            for i in range(l):
                cols[i].append(record[i])
        batch = []
        for col in cols:
            col = np.array(col)
            if len(col.shape) == 1:
                 col = np.expand_dims(col, axis=1)
            batch.append(col)
        return batch


    def get_sample_weights(self):
        return self.sample_weights

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
        self.sample_weights = weights.astype(np.float32)


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
        record = Record([i,i], i, i, [0,0], i % 2 == 0, np.array([i,i]))
        rb.add(record)


    print(rb.size())
    sample = rb.sample_batch(2)
    for s in sample:
        print(s)

