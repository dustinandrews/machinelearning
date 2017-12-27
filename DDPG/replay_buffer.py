# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:56:40 2017

@author: dandrews
"""

from collections import deque
import random
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
        else:
            self.buffer = deque()

    def add(self, s, a, r, t, s2, qe):
        experience = (s, a, r, t, s2, qe)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size: int):
        batch = []
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)
        return self.get_batches_from_list(batch)


    def to_batches(self):
       return self.get_batches_from_list(self.buffer)

    def get_batches_from_list(self, batch):
        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])
        qe_batch = np.array([_[5] for _ in batch])
        return s_batch, a_batch, r_batch, t_batch, s2_batch, qe_batch



    def clear(self):
      self.deque.clear()
      self.count = 0

if __name__ == '__main__':
    r = ReplayBuffer(10)
#    r.add([1,1], 1, 0, False, [1,0])
#    r.add([1,1], 1, -1, True, [1,0])
    print(r.size())
    print(r.sample_batch(2))

