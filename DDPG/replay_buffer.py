# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:56:40 2017

@author: dandrews
"""

from collections import deque
import random
import numpy as np

class ReplayBuffer(object):
    
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        
    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
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
    
    
    def sample_worst_batch(self, batch_size:int):
        batch = []
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            for record in self.buffer:
                if record[2][0] < 0:
                    batch.append(record)
            batch = random.sample(batch, batch_size)
        
        return self.get_batches_from_list(batch)
            
   

    def get_batches_from_list(self, batch):
        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])
        return s_batch, a_batch, r_batch, t_batch, s2_batch
     
            

    def clear(self):
      self.deque.clear()
      self.count = 0
      
if __name__ == '__main__':
    r = ReplayBuffer(10)
    r.add([1,1], 1, 0, False, [1,0])
    r.add([1,1], 1, -1, True, [1,0])
    print(r.size())
    print(r.sample_batch(2))
    print(r.sample_worst_batch(1))
    
          