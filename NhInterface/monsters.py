# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 16:46:41 2018

@author: dandrews
"""

import numpy as np

class Monsters():
    """
    List of monster glyphs and normalized data associated with them.
    """
    def __init__(self, glyphs):
        self.names, self.monster_data = self.normalize_monster_data(glyphs)
        self.data_names = ['level', 'monstr', 'move', 'ac']
        gkeys = np.array([k for k in glyphs if glyphs[k]['type'] == 'monster'], dtype=np.int)
        self.minkey = gkeys.min()
        self.maxkey = gkeys.max()

    #def is_monster(glyph: int):


    def normalize_monster_data(self, glyphs):
        ac = []
        level = []
        monstr = []
        move = []
        name = []
        for key in sorted(glyphs):
            g_type = glyphs[key]['type']
            if g_type == 'monster':
                data = glyphs[key]['data']
                ac.append(data['ac'])
                level.append(data['level'])
                monstr.append(data['monstr'])
                move.append(data['move'])
                name.append(data['name'])

        ac = np.array(ac, dtype=np.float32)
        level = np.array(level, dtype=np.float32)
        monstr = np.array(monstr, dtype=np.float32)
        move = np.array(move, dtype=np.float32)

        ac *= -1 # Good AC in game is negative
        _ac = self.normalize_np_array(ac)
        _level = self.normalize_np_array(level)
        _monstr = self.normalize_np_array(monstr)
        _move = self.normalize_np_array(move)

        ret_data = []
        for i in range(len(ac)):
            ret_data.append([
                    _level[i],
                    _monstr[i],
                    _move[i],
                    _ac[i]
                    ])
        ret_data = np.array(ret_data, dtype=np.float32)
        return name, ret_data

    def normalize_np_array(self, np_arr):
        a_min = np_arr.min()
        np_arr -= a_min
        a_max = np_arr.max()
        np_arr /= a_max
        return np_arr

if __name__ == '__main__':
    import pickle
    glyph_pickle_file = "glyphs.pkl"

    with open(glyph_pickle_file, 'rb') as glyph_file:
        glyphs = pickle.load(glyph_file)

    mons = Monsters(glyphs)