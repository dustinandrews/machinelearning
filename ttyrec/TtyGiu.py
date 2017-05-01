# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 20:43:25 2017

@author: dandrews
"""
from tkinter import Tk, Text, INSERT, END, Button
import tkinter.font as tkFont
from TtyParse import TtyParse

class TtyRender:
    self.parser = None
    self.screen = None

    def __init__(self):
        self.fg = 'gray85'
        self.bg = 'black'
        self.tag = 0
        self.frame = 0
        self.parser = TtyParse(glob.glob('./*/*/*.ttyrec')[0])
        self.parser.get_metadata()
        self.parser.render_frames(0,0)
        width = self.parser.metadata.collumns
        height = self.parser.metadata.lines

        root = Tk()
        self.customFont = tkFont.Font(family="Lucida Console", size=12)
        self.text = Text(root, font=self.customFont, width=width, height=height, background=self.bg, foreground=self.fg) #fg='gray94', bg='black'
        self.show_text()      
        self.text.pack()
        b = Button(text="next", command=self.callback)
        b.pack()
        root.mainloop()
        
    def callback(self):
        self.parser.render_frames(self.frame, self.frame + 1 )
        self.frame += 1
        self.show_text()
        pass
    
    def show_text(self):
        self.text.delete('1.0', END)
        for line in self.parser.screen.buffer:
            for c in line:
                if self.get_tag_from_char(c):
                    self.text.insert(INSERT, c.data, (str(self.tag)))
                else:
                    self.text.insert(INSERT, c.data)
            self.text.insert(INSERT, "\n")
        
    def get_tag_from_char(self, c):
        if c.fg != 'default' or c.bg !='default' or c.bold or c.italics or c.underscore or c.strikethrough or c.reverse:
            if c.fg == 'default':
                fg = self.fg
            else:
                fg = c.fg
            
            if c.bg == 'default':
                bg = self.bg
            else:
                bg = c.fg
            self.tag += 1
            if c.reverse:
                self.text.tag_config(str(self.tag), background=fg, foreground=bg, underline = c.underscore)
            else:
                self.text.tag_config(str(self.tag), background=bg, foreground=fg, underline = c.underscore)
            return True
        else:
            return False;


if __name__ == "__main__":
    tr = TtyRender()