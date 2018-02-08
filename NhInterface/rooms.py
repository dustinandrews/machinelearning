# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 10:54:40 2018

@author: dandrews
"""
import numpy as np

class RoomTiles():

        # 846 is iron bars, effectively a wall in most cases.
        # 847 tree
        # Pick highest value to make walls "solid"
    WALL_GLYPHS = [1056, 830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840,
                   846, 847,
                   1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022,
                   1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032,
                   1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042,
                   1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052,
                   1053, 1054, 1055
                   ]
    FLOOR_GLYPHS = [848, 849, 850, 829]
    DOOR_GLYPHS_CLOSED = [844,845]
    DOOR_GLYPHS_OPENED = [842, 843]
    UP_GLYPHS = [851, 853]
    DOWN_GLYPHS = [852, 854]
    DB_LOWERED = [863, 864]
    DB_RAISED = [865, 866]

    # Things to treat collapse into architypes
    GLYPH_COLLECTIONS = [WALL_GLYPHS, FLOOR_GLYPHS, DOOR_GLYPHS_OPENED,
                         DOOR_GLYPHS_CLOSED, UP_GLYPHS, DOWN_GLYPHS]

    _min_trap = 870
    _max_trap = 892
    _swallow = [i for i in range(905, 921)]

    def __init__(self, glyphs):
        room_keys = np.array([k for k in glyphs if glyphs[k]['type'] == 'room'], dtype=np.int)
        self.minkey = room_keys.min()
        self.maxkey = np.max(list(glyphs.keys()))

        trap_keys = [i for i in range(self._min_trap, self._max_trap + 1)]
        self._room_data = self._normalize_room_data(room_keys, trap_keys)
        self._room_data = self._compact_room_data(self._room_data)



    def collapse_glyph(self, glyph):
        """
        Converts equivalent classes of glyphs to the first of the type
        For example all walls are converted to just one wall
        """
        for glist in self.GLYPH_COLLECTIONS:
            if glyph in glist:
                glyph = glist[0]
        return glyph

    def _normalize_room_data(self, room_keys, trap_keys):
        # more traps than rooms types, keep the arrays even by using traps
        num_traps = len(trap_keys)
        trap_eye = np.eye(num_traps)

        unique_rooms = []
        #count unique types
        for i in range(np.min(room_keys), np.min(trap_keys)):
            u = self.collapse_glyph(i)
            if u not in unique_rooms:
                unique_rooms.append(u)

        one_hots = {}
        for i in range(np.min(room_keys), np.min(trap_keys)):
            u = self.collapse_glyph(i)
            index = unique_rooms.index(u)
            one_hots[i] = [trap_eye[index], [0] * num_traps]

        mt = np.min(trap_keys)
        for i in range(mt, self._max_trap + 1):
            one_hots[i] = [[0] * num_traps, trap_eye[i - mt] ]

        return one_hots

    def _compact_room_data(self, room_dict: dict):
        out_arr = []
        for i in sorted(room_dict.keys()):
            out_arr.append(room_dict[i])

        return np.array(out_arr, dtype=np.float32)


    def get_room_data(self, glyph_num: int):
        if glyph_num < self.minkey or glyph_num > self.maxkey :
            raise ValueError("{} outside rooms range of  {} - {}.".format(glyph_num, self.minkey, self.maxkey))

        if glyph_num > self._max_trap:
            return self._room_data[self._max_trap - self.minkey]
        index = glyph_num - self.minkey
        return self._room_data[index]

if __name__ == '__main__':
    import pickle
    glyph_pickle_file = "glyphs.pkl"

    with open(glyph_pickle_file, 'rb') as glyph_file:
        glyphs = pickle.load(glyph_file)

    rt = RoomTiles(glyphs)

