# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 13:50:11 2018

@author: dandrews based on lmj.nethack
"""
import collections
import pickle

class NhData():
    """
    A clean place to store some facts about the game
    """
    # Name: Human discriptions
    # Command: string to send to game
    # Rating: subjective assesment of how hard the command is to use
    Command = collections.namedtuple('command', 'name command rating')

    WALL_GLYPHS = [830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 840,
                   1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022,
                   1023, 1024, 1025, 1026, 1027, 1028, 1029, 1030, 1031, 1032,
                   1033, 1034, 1035, 1036, 1037, 1038, 1039, 1040, 1041, 1042,
                   1043, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052,
                   1053, 1054, 1055, 1056
                   ]

    FLOOR_GLYPHS = [848, 849, 850]

    DOOR_GLYPHS = [845,846]

    GLYPH_COLLECTIONS = [WALL_GLYPHS, FLOOR_GLYPHS, DOOR_GLYPHS]

    COMMANDS = {
        0: Command ('UP', '<', 1 ),
        1: Command ('DOWN', '>', 1 ),

        # numpad version, otherwise lkuyjnb
        2: Command ('E', '6', 1 ),
        3: Command ('N', '8', 1 ),
        4: Command ('NE', '9', 1 ),
        5: Command ('NW', '7', 1 ),
        6: Command ('S', '2', 1 ),
        7: Command ('SE', '3', 1 ),
        8: Command ('SW', '1', 1 ),
        9: Command ('W', 'h', 1 ),

        10:Command('WAIT', '.', 2 ),
        11:Command('OPEN', 'o', 10 ),
        12:Command('PICKUP', ',', 2 ),
        13:Command('APPLY', 'a', 10 ),
        14:Command('CAST', 'Z', 10 ),
        15:Command('CLOSE', 'c', 10 ),
        16:Command('DROP', 'd', 2 ),
        17:Command('EAT', 'e', 3 ),
        18:Command('EXCHANGE', 'x', 10 ),
        19:Command('FIRE', 'f', 10 ),
        20:Command('INVENTORY', 'i', 10 ),
        21:Command('KICK', '\x03', 2 ),
        22:Command('MORE', '\r', 0 ), # Probably not used by agents
        23:Command('PAY', 'p', 10 ),
        24:Command('PUTON', 'P', 10 ),
        25:Command('QUAFF', 'q', 3 ),
        26:Command('QUIVER', 'Q', 10 ),
        27:Command('READ', 'r', 10 ),
        28:Command('REMOVE', 'R', 10 ),
        29:Command('SEARCH', 's', 2 ),
        30:Command('TAKEOFF', 'T', 10 ),
        31:Command('TELEPORT', '\x14', 10 ),
        32:Command('THROW', 't', 10 ),
        33:Command('WEAR', 'W', 10 ),
        34:Command('WIELD', 'w', 10 ),
        35:Command('ZAP', 'z', 10 ),
        36:Command('ENGRAVE', 'E', 10 ),
        37:Command('CHAT', '#chat', 10 ),
        38:Command('DIP', '#dip', 10 ),
        39:Command('FORCE', '#force', 10 ),
        40:Command('INVOKE', '#invoke', 10 ),
        41:Command('JUMP', '#jump', 10 ),
        42:Command('LOOT', '#loot', 10 ),
        43:Command('MONSTER', '#monster', 10 ),
        44:Command('OFFER', '#offer', 10 ),
        45:Command('PRAY', '#pray', 2 ),
        46:Command('RIDE', '#ride', 10 ),
        47:Command('RUB', '#rub', 10 ),
        48:Command('SIT', '#sit', 10 ),
        49:Command('TURN', '#turn', 10 ),
        50:Command('WIPE', '#wipe', 10)
        }

    #Sample telnet data for testing
    SAMPLE_DATA = [b' \x1b[H\x1b[2J\x1b[2d ## \x1b(B\x1b[0;1m\x1b[33m\x1b[40mnethack.alt.org - http://nethack.alt.org/\x1b[6;2H\x1b[39;49m\x1b(B\x1b[mPlease enter your username. (blank entry aborts)\r\x1b[8d =>',
 b' aa\x1b[H\x1b[2J\x1b[2d ## \x1b(B\x1b[0;1m\x1b[33m\x1b[40mnethack.alt.org - http://nethack.alt.org/\x1b[6;2H\x1b[39;49m\x1b(B\x1b[mPlease enter your password.\r\x1b[8d =>',
 b' \x1b[H\x1b[2J\x1b[2d  ## \x1b(B\x1b[0;1m\x1b[33m\x1b[40mnethack.alt.org - http://nethack.alt.org/\x1b[3;3H\x1b[39;49m\x1b(B\x1b[m##\x1b[4d\x08\x08## dgamelaunch 1.5.1 - network console game launcher\x1b[5;3H## Copyright (c) 2000-2010 The Dgamelaunch Team\x1b[6;3H## See http://nethackwiki.com/wiki/dgamelaunch for more info\x1b[7;3H##\x1b[8d\x08\x08## Games on this server are recorded for in-progress viewing and playback!\x1b[10;3HLogged in as: aa\x1b[12;3Hc) Change password\x1b[13;3He) Change email address\x1b[14;3Hw) Watch games in progress\x1b[15;3Ho) Edit options\x1b[16;3Hp) Play NetHack 3.6.1\x1b[17;3Hq) Quit\x1b[19;3H=>',
 b'\x1b[H\x1b[2J\x1b[24;1H\x1b[?1049l\r\x1b[?1l\x1b>\x1b[?1049h\x1b[?1l\x1b>\x1b[2;0z\x1b[2J\x1b[H\x1b[2;1HNetHack, Copyright 1985-2003\r\x1b[3;1H         By Stichting Mathematisch Centrum and M. Stephenson.\r\x1b[4;1H         See license for details.\r\x1b[5;1H\x1b[2;2z\x1b[2;1z\x1b[2;3z\r\n\x1b[2;1z\x1b[HRestoring save file...\x1b[K\x1b[2;0z\x1b[7m--More--\x1b[27m\x1b[3z',
 b'\x1b[2;0z\x1b[2;1z\x1b[H\x1b[K\x1b[2;3z\x1b[2J\x1b[H\x1b[2;1z\x1b[2;3z\x1b[3;30H\x1b[0;832z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;833z-\x1b[1z\x1b[0m\x1b[4;30H\x1b[0;830z|\x1b[1z\x1b[0;848z\x1b[0m\x1b[1m\x1b[30m.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;844z\x1b[0m\x1b[1m\x1b[31m+\x1b[1z\x1b[0m\x1b[4;69H\x1b[0;832z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;833z-\x1b[1z\x1b[0m\x1b[5;30H\x1b[0;830z|\x1b[1z\x1b[0;848z\x1b[0m\x1b[1m\x1b[30m.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;830z\x1b[0m|\x1b[1z\x1b[0m\x1b[5;45H\x1b[0;832z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;833z-\x1b[1z\x1b[0m\x1b[5;69H\x1b[0;830z|\x1b[1z\x1b[0;848z\x1b[0m\x1b[1m\x1b[30m.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;830z\x1b[0m|\x1b[1z\x1b[0m\x1b[6;30H\x1b[0;830z|\x1b[1z\x1b[0;848z\x1b[0m\x1b[1m\x1b[30m.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;830z\x1b[0m|\x1b[1z\x1b[0m\x1b[C\x1b[C\x1b[0;849z\x1b[34m#\x1b[1z\x1b[0m\x1b[C\x1b[C\x1b[0;849z\x1b[34m#\x1b[1z\x1b[0;842z\x1b[0m\x1b[1m\x1b[31m-\x1b[1z\x1b[0;848z\x1b[0m\x1b[1m\x1b[30m.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;859z\x1b[0m\x1b[1m\x1b[34m{\x1b[1z\x1b[0;852z\x1b[0m\x1b[1m\x1b[35m>\x1b[1z\x1b[0;848z\x1b[0m\x1b[1m\x1b[30m.\x1b[1z\x1b[0;841z\x1b[0m\x1b[1m\x1b[31m.\x1b[1z\x1b[0;849z\x1b[0m\x1b[34m#\x1b[1z\x1b[0m\x1b[6;62H\x1b[0;841z\x1b[1m\x1b[31m.\x1b[1z\x1b[0m\x1b[6;69H\x1b[0;830z|\x1b[1z\x1b[0;851z\x1b[0m\x1b[1m\x1b[35m<\x1b[1z\x1b[0;848z\x1b[0m\x1b[1m\x1b[30m.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;830z\x1b[0m|\x1b[1z\x1b[0m\x1b[7;30H\x1b[0;830z|\x1b[1z\x1b[0;848z\x1b[0m\x1b[1m\x1b[30m.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;842z\x1b[0m\x1b[1m\x1b[31m-\x1b[1z\x1b[0;849z\x1b[0m\x1b[34m#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;830z\x1b[0m|\x1b[1z\x1b[0;848z\x1b[0m\x1b[1m\x1b[30m.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;858z\x1b[0m\x1b[1m\x1b[34m#\x1b[1z\x1b[0;848z\x1b[0m\x1b[1m\x1b[30m.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;830z\x1b[0m|\x1b[1z\x1b[0;849z\x1b[0m\x1b[34m#\x1b[1z\x1b[0m\x1b[7;69H\x1b[0;844z\x1b[1m\x1b[31m+\x1b[1z\x1b[0;848z\x1b[0m\x1b[1m\x1b[30m.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;830z\x1b[0m|\x1b[1z\x1b[0m\x1b[8;30H\x1b[0;834z-\x1b[1z\x1b[0;841z\x1b[0m\x1b[1m\x1b[31m.\x1b[1z\x1b[0;831z\x1b[0m-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;835z-\x1b[1z\x1b[0m\x1b[8;43H\x1b[0;849z\x1b[34m#\x1b[1z\x1b[0m\x1b[C\x1b[0;834z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;835z-\x1b[1z\x1b[0;849z\x1b[0m\x1b[34m#\x1b[1z\x1b[0m\x1b[8;61H\x1b[0;845z\x1b[1m\x1b[31m+\x1b[1z\x1b[0m\x1b[C\x1b[0;849z\x1b[34m#\x1b[1z\x1b[0m\x1b[8;69H\x1b[0;830z|\x1b[1z\x1b[0;848z\x1b[0m\x1b[1m\x1b[30m.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;830z\x1b[0m|\x1b[1z\x1b[0m\x1b[9;31H\x1b[0;849z\x1b[34m#\x1b[1z\x1b[0m\x1b[C\x1b[0;849z\x1b[34m#\x1b[1z\x1b[0m\x1b[9;51H\x1b[0;849z\x1b[34m#\x1b[1z\x1b[0;823z\x1b[0m0\x1b[1z\x1b[0;849z\x1b[0m\x1b[34m#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;841z\x1b[0m\x1b[1m\x1b[31m.\x1b[1z\x1b[0;848z\x1b[0m\x1b[1m\x1b[30m.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;830z\x1b[0m|\x1b[1z\x1b[0m\x1b[10;31H\x1b[0;849z\x1b[34m#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0m\x1b[10;52H\x1b[0;849z\x1b[34m#\x1b[1z\x1b[0m\x1b[10;60H\x1b[0;849z\x1b[34m#\x1b[1z\x1b[0m\x1b[C\x1b[C\x1b[0;849z\x1b[34m#\x1b[1z\x1b[0m\x1b[10;69H\x1b[0;830z|\x1b[1z\x1b[0;848z\x1b[0m\x1b[1m\x1b[30m.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;830z\x1b[0m|\x1b[1z\x1b[0m\x1b[11;31H\x1b[0;849z\x1b[34m#\x1b[1z\x1b[0m\x1b[11;58H\x1b[0;849z\x1b[34m#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0m\x1b[C\x1b[C\x1b[0;849z\x1b[34m#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0m\x1b[C\x1b[C\x1b[C\x1b[0;834z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;835z-\x1b[1z\x1b[0m\x1b[12;31H\x1b[0;849z\x1b[34m#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0m\x1b[12;52H\x1b[0;849z\x1b[34m#\x1b[1z\x1b[0m\x1b[12;58H\x1b[0;849z\x1b[34m#\x1b[1z\x1b[0m\x1b[12;65H\x1b[0;849z\x1b[34m#\x1b[1z\x1b[0m\x1b[13;31H\x1b[0;849z\x1b[34m#\x1b[1z\x1b[0m\x1b[C\x1b[0;849z\x1b[34m#\x1b[1z\x1b[0m\x1b[13;49H\x1b[0;849z\x1b[34m#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0m\x1b[C\x1b[0;849z\x1b[34m#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0m\x1b[14;33H\x1b[0;849z\x1b[34m#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0m\x1b[14;49H\x1b[0;849z\x1b[34m#\x1b[1z\x1b[0;832z\x1b[0m-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;841z\x1b[0m\x1b[1m\x1b[31m.\x1b[1z\x1b[0;831z\x1b[0m-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;833z-\x1b[1z\x1b[0;849z\x1b[0m\x1b[34m#\x1b[1z\x1b[0m\x1b[C\x1b[C\x1b[C\x1b[0;849z\x1b[34m#\x1b[1z\x1b[0m\x1b[15;21H\x1b[0;832z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;845z\x1b[0m\x1b[1m\x1b[31m+\x1b[1z\x1b[0;831z\x1b[0m-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;841z\x1b[0m\x1b[1m\x1b[31m.\x1b[1z\x1b[0;831z\x1b[0m-\x1b[1z\x1b[0;833z-\x1b[1z\x1b[0m\x1b[15;49H\x1b[0;849z\x1b[34m#\x1b[1z\x1b[0;830z\x1b[0m|\x1b[1z\x1b[0;848z\x1b[0m\x1b[1m\x1b[30m.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;830z\x1b[0m|\x1b[1z\x1b[0;849z\x1b[0m\x1b[34m#\x1b[1z\x1b[0m\x1b[C\x1b[C\x1b[C\x1b[0;849z\x1b[34m#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0m\x1b[16;21H\x1b[0;841z\x1b[1m\x1b[31m.\x1b[1z\x1b[0;848z\x1b[0m\x1b[1m\x1b[30m.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;830z\x1b[0m|\x1b[1z\x1b[0m\x1b[16;49H\x1b[0;849z\x1b[34m#\x1b[1z\x1b[0;830z\x1b[0m|\x1b[1z\x1b[0;848z\x1b[0m\x1b[1m\x1b[30m.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;830z\x1b[0m|\x1b[1z\x1b[0;849z\x1b[0m\x1b[34m#\x1b[1z\x1b[0m\x1b[16;69H\x1b[0;849z\x1b[34m#\x1b[1z\x1b[0m\x1b[17;21H\x1b[0;830z|\x1b[1z\x1b[0;848z\x1b[0m\x1b[1m\x1b[30m.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;830z\x1b[0m|\x1b[1z\x1b[0m\x1b[17;49H\x1b[0;849z\x1b[34m#\x1b[1z\x1b[0;830z\x1b[0m|\x1b[1z\x1b[0;848z\x1b[0m\x1b[1m\x1b[30m.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;830z\x1b[0m|\x1b[1z\x1b[0;849z\x1b[0m\x1b[34m#\x1b[1z\x1b[0m\x1b[17;69H\x1b[0;849z\x1b[34m#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;832z\x1b[0m-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;833z-\x1b[1z\x1b[0m\x1b[18;21H\x1b[0;830z|\x1b[1z\x1b[0;848z\x1b[0m\x1b[1m\x1b[30m.\x1b[1z\x1b[0;16z\x1b[0m\x1b[1m\x1b[37m\x1b[7md\x1b[0m\x1b[0m\x1b[1z\x1b[0;848z\x1b[1m\x1b[30m.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;830z\x1b[0m|\x1b[1z\x1b[0m\x1b[18;49H\x1b[0;849z\x1b[34m#\x1b[1z\x1b[0;830z\x1b[0m|\x1b[1z\x1b[0;848z\x1b[0m\x1b[1m\x1b[30m.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;830z\x1b[0m|\x1b[1z\x1b[0;849z\x1b[0m\x1b[34m#\x1b[1z\x1b[0m\x1b[18;71H\x1b[0;849z\x1b[34m#\x1b[1z\x1b[0;841z\x1b[0m\x1b[1m\x1b[31m.\x1b[1z\x1b[0;848z\x1b[0m\x1b[1m\x1b[30m.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;830z\x1b[0m|\x1b[1z\x1b[0m\x1b[19;21H\x1b[0;830z|\x1b[1z\x1b[0;45z\x1b[0m\x1b[1m\x1b[37mh\x1b[1z\x1b[0;848z\x1b[0m\x1b[1m\x1b[30m.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;830z\x1b[0m|\x1b[1z\x1b[0m\x1b[19;50H\x1b[0;830z|\x1b[1z\x1b[0;786z\x1b[0m\x1b[1m\x1b[33m$\x1b[1z\x1b[0;848z\x1b[0m\x1b[1m\x1b[30m.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;830z\x1b[0m|\x1b[1z\x1b[0;849z\x1b[0m\x1b[34m#\x1b[1z\x1b[0m\x1b[19;72H\x1b[0;830z|\x1b[1z\x1b[0;848z\x1b[0m\x1b[1m\x1b[30m.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;830z\x1b[0m|\x1b[1z\x1b[0m\x1b[20;21H\x1b[0;834z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;835z-\x1b[1z\x1b[0m\x1b[20;50H\x1b[0;834z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;835z-\x1b[1z\x1b[0;849z\x1b[0m\x1b[34m#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;849z#\x1b[1z\x1b[0;841z\x1b[0m\x1b[1m\x1b[31m.\x1b[1z\x1b[0;848z\x1b[0m\x1b[1m\x1b[30m.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;830z\x1b[0m|\x1b[1z\x1b[0m\x1b[21;72H\x1b[0;834z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;835z-\x1b[1z\x1b[0m\x1b[19;22H\x1b[2;2z\x1b[23;1H\x1b[K[\x1b[7m\x08\x1b[1m\x1b[32m\x1b[CAa the Stripling\x1b[0m\x1b[0m\x1b[0m\r\x1b[23;18H]          St:18/02 Dx:14 Co:16 In:8 Wi:9 Ch:8  Lawful S:46\r\x1b[24;1HDlvl:1  $:30 HP:\x1b[K\r\x1b[1m\x1b[32m\x1b[24;17H18(18)\x1b[0m\r\x1b[24;23H Pw:\r\x1b[1m\x1b[32m\x1b[24;27H1(1)\x1b[0m\r\x1b[24;31H AC:6  Xp:1/4 T:194\x1b[2;1z\x1b[HVelkommen aa, the dwarven Valkyrie, welcome back to NetHack!\x1b[K\x1b[2;3z\x1b[19;22H\x1b[3z']

# Maybe group glyphs by type? (first,last)
#    CATEGORIES = {'Amulets': (), 'Weapons': (), 'Armor': (), 'Comestibles': (),
#    'Scrolls': (), 'Spellbooks: ()', 'Potions: ()', 'Rings: ()',
#    'Wands':(), 'Tools':(), 'Gems':()}

    def __init__(self):
        self.monster_data = pickle.load(open( "nh_monster.p", "rb" ) )

    def collapse_glyph(self, glyph):
        """
        Converts equivalent classes of glyphs to the first of the type
        """
        for glist in self.GLYPH_COLLECTIONS:
            if glyph in glist:
                glyph = glist[0]
        return glyph

if __name__ =='__main__':
    nhd = NhData()
