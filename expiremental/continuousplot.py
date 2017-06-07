# -*- coding: utf-8 -*-
"""
Created on Wed May 31 16:42:42 2017

@author: dandrews
"""

import numpy as np
import matplotlib.pyplot as plt

#plt.axis([0, 10, 0, 1])
plt.ion()
ax = plt.gca()
ax.set_yscale('log')
while True:
    y = []
    for i in range(10):
        y.append(np.random.random())
    s = plt.plot(range(10), y)
    plt.pause(0.5)
    s.remove()