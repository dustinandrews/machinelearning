# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 17:19:51 2017

@author: dandrews
"""

from textmap import Map

with open("textmap.cntk.txt", "w") as f:
    for i in range(100000):
        m = Map(5,5)
        f.write("|features ")
        f.write(" ".join(str(d) for d in m.data()))
        f.write("|labels ")
        f.write(" ".join(str(l) for l in m.labels()))
        f.write("|xy ")
        f.write(" ".join(str(xy) for xy in m.xy_data()))
        f.write("\n")

with open("textmap.cntk.txt", "r") as f:
    row = f.readline()
    print(row)
    
    