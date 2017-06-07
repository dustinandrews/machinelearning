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
plt.ion()
ax = plt.gca()
#ax.set_yscale('log')
ax.set_xbound(0,255)
ax.set_ybound(-15000,15000)


buffer_size = 1024 * 8

path = r'E:\wingame\e\57833-1_v300_dev__destiny_v300_silver_w64'

file = glob.glob(path + r'\*.input_stream')[0]
print(file)
baseline = np.zeros(256)
plt.title("Getting baseline")
index = 0
counts = np.zeros(256)
barx = range(256)
bars = []
while True:
    t = [b.remove() for b in bars]
    size = os.path.getsize(file)   
    end = int(((size/buffer_size) -1) * buffer_size)
    counts -= counts
    with open(file, "rb") as f:
        f.seek(end, 1)
        buffer = f.read(buffer_size)
        for b in buffer:
            counts[b] +=1
            if index < 10:
                baseline[b] += 1
    counts[0] = 0 # irrelevant and noisy
    baseline[0] = 0
    counts[255] = 0 # irrelevant and noisy
    baseline[255] = 0
    y = counts
    
    if index < 10:
        plt.title("Getting baseline {}/10".format(index))
    
    if index == 10:        
        print(baseline[0])
        baseline /= 10
        plt.title("Counts - Baseline")
        print(baseline[0])
        y -= baseline




    if index > 10:
        print(np.argmax(y))
        y -= baseline
        print(np.argmax(y))
        
    bars = plt.bar(barx, y, color='b')
    ax = plt.gca()
    ax.relim()
    ax.autoscale_view()        

    #count, bins, bars = plt.hist(y, 255, range=[0, 255], color='b')
    #s.remove()
    plt.pause(0.01)
    
    index += 1
    
        
