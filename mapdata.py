# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 17:19:51 2017

@author: dandrews
"""

from textmap import Map

with open("textmap.cntk.txt", "w") as f:
    for i in range(100000):
        m = Map(10,10)
        f.write("|features ")
        f.write(" ".join(str(d) for d in m.data()))
        f.write("|labels ")
        f.write(" ".join(str(l) for l in m.labels()))
        f.write("\n")