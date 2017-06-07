# -*- coding: utf-8 -*-
"""
Created on Wed May 31 15:17:37 2017

@author: dandrews
"""

import glob
import os
from matplotlib import pyplot as plt
import numpy as np

#plt.axis([0, 10, 0, 1])
#plt.ion()
ax = plt.gca()
#ax.set_yscale('log')
ax.set_xbound(0,255)
ax.set_ybound(-15000,15000)


path = r'E:\wingame\e\57833-1_v300_dev__destiny_v300_silver_w64'

file = glob.glob(path + r'\down.input_stream')[0]
print(file)
baseline = np.zeros(256)
plt.title("down")
index = 0
counts = np.zeros(256)
barx = range(256)
bars = []

size = os.path.getsize(file)   
with open(file, "rb") as f:        
    buffer = f.read(size)
    for b in buffer:
        counts[b] +=1
y = counts
y[0] = 0
        
bars = plt.bar(barx, y, color='b')
ax = plt.gca()
ax.relim()
ax.autoscale_view()        

plt.show()   
