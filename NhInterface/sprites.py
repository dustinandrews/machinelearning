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


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ss = SpriteSheet("chozo32.bmp",40,30)
    img = ss.get_image(0,0,32,32)
    plt.imshow(img)
    plt.show()
    img = ss.get_image_by_number(30)
    plt.imshow(img)
    plt.show()

#    for i in range(2):
#        img = ss.get_image_by_number(i)
#        plt.imshow(img)
#        plt.show()
#
#    for i in range(1,3):
#        img = ss.get_image_by_number(i*40)
#        plt.imshow(img)
#        plt.show()

