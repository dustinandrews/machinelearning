# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:47:51 2018

@author: dandrews
"""
import numpy as np

class Objects():
    object_classes = {
            1: 'strange object',
            2: 'weapon',
            3: 'armor',
            4: 'ring',
            5: 'amulet',
            6: 'container',
            7: 'food',
            8: 'potion',
            9: 'scroll',
            10: 'spell book',
            11: 'wand',
            12: 'gold',
            13: 'gem',
            14: 'boulder/statue',
            15: 'iron ball',
            16: 'iron chain',
            17: 'venom',
            }

    def __init__(self, glyphs):
        self.glyphs = glyphs
        gkeys = np.array([k for k in glyphs if glyphs[k]['type'] == 'object'], dtype=np.int)
        self.minkey = gkeys.min()
        self.maxkey = gkeys.max()

    def get_object(self, index: int):
        obj_type = self.glyphs[index]['type']
        if obj_type != 'object':
            raise ValueError("Index {} is type '{}', not 'object' ".format(index, obj_type))
        d = self.glyphs[index]['data']

        obj = {'name':d['name'], 'class':d['class'], 'info':self.object_classes[d['class']]}
        return obj