# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 13:47:08 2018

@author: http://programarcadegames.com/python_examples/en/sprite_sheets/
This module is used to pull individual sprites from sprite sheets.
"""
import scipy.ndimage

class SpriteSheet(object):
    """ Class used to grab images out of a sprite sheet. """

    def __init__(self, file_name, rows, columns):
        """ Constructor. Pass in the file name of the sprite sheet. """
        self.rows = rows
        self.columns = columns

        self.sheet = scipy.ndimage.imread(file_name)
#        self.sprite_h = self.sheet.shape[0] // columns
#        self.sprite_w = self.sheet.shape[1] // rows
        self.sprite_h = 32
        self.sprite_w = 32

    def get_image_by_number(self, num):
        row = num // self.rows
        col = num - (row * self.rows)
        top = row * self.sprite_h
        left = col * self.sprite_w
        #print("row:{}, col:{}, glyph:{}".format(row, col, num))
        image = self.get_image(top, left,self.sprite_h, self.sprite_w )
        return image


    def get_image(self, x, y, width, height):
        """ Grab a single image out of a larger spritesheet
            Pass in the x, y location of the sprite
            and the width and height of the sprite. """

        image = self.sheet[x:x+width,y:y+width,:]
        return image

    def plot_small_glyph(self, glyph_num):
        fig = plt.figure(frameon=False)
        fig.set_size_inches(0.5,0.5)
        ax = fig.add_axes([0,0,1,1])
        ax.axis('off')
        ax.imshow(self.get_image_by_number(glyph_num))
        plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ss = SpriteSheet("sprite_sheets/chozo32.bmp",40,30)
    ss.plot_small_glyph(431)

