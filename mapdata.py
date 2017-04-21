# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 17:19:51 2017

@author: dandrews
"""

from textmap import Map
import numpy as np

with open("textmap.cntk.txt", "w") as f:
    for i in range(100000):
        m = Map(5,5)
        f.write("|features ")
        f.write(" ".join(str(d) for d in m.data()))
        f.write("|labels ")
        f.write(" ".join(str(l) for l in m.labels()))
        f.write("\n")

with open("textmap.cntk.txt", "r") as f:
    row = f.readline()
    row = row.replace("|features ", "")
    row = row.replace("|labels", "")
    items = row.split(" ")
    nums = [float(i) for i in items[:100]]
    npnums = np.array(nums, dtype=np.float32)
    print(npnums)
    
m = Map(5,5)
m.load_from_data(npnums)
m.display()
    