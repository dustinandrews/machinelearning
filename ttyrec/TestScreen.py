# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 14:17:11 2017

@author: dandrews
"""
from pyte import Screen

#import colorama
cursor_home = '\033[1;1H'

class TestScreen(Screen):
    """
    Override the screen class with a screen
    that can detect the size based on sample data
    Initialize to the biggest size you want to handle.
    """
    subclass = True
    maxline = 0
    maxcolumn = 0
    def __init__(self, columns, lines):
        super(TestScreen, self).__init__(columns, lines)
        
    def draw(self, char):
        super(TestScreen, self).draw(char)
        if self.cursor.x > self.maxcolumn:
            self.maxcolumn = self.cursor.x
        
        if self.cursor.y > self.maxline:
            self.maxline = self.cursor.y