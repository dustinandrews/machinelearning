# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 14:17:11 2017

@author: dandrews
"""
from pyte import Screen, screens

#import colorama
cursor_home = '\033[1;1H'

class TestScreen(Screen):
    """
    Override the screen class with a screen
    that can detect the size based on sample data
    Initialize to the biggest size you want to handle.
    
    "fg",
    "bg",
    "bold",
    "italics",
    "underscore",
    "strikethrough",
    "reverse",
    
    """
    subclass = True
    maxline = 0
    maxcolumn = 0
    style_list = {}
    def __init__(self, columns, lines):
        super(TestScreen, self).__init__(columns, lines)
        
    def draw(self, char):
        super(TestScreen, self).draw(char)
        if self.cursor.x > self.maxcolumn:
            self.maxcolumn = self.cursor.x
        
        if self.cursor.y > self.maxline:
            self.maxline = self.cursor.y
        
#        buff = self.buffer[self.cursor.y][self.cursor.x]
#        style = "'" + buff.data +"'" \
#        + " fg: " + str(buff.fg) \
#        + " bg: " + str(buff.bg) \
#        + " Bold: " + str(buff.bold) \
#        + " Italics: " + str(buff.italics) \
#        + " Underscore: " + str(buff.underscore) \
#        + " Strikethough: " + str(buff.strikethrough) \
#        + " Reverse: " + str(buff.reverse)
#        
#        if style in self.style_list:
#            self.style_list[style] += 1
#        else:
#            self.style_list[style] = 1
        