# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 09:17:45 2017

@author: dandrews
"""
import win32ui
import win32gui
import win32con
import numpy as np
from matplotlib import pyplot as plt

class TigerScreen:
    hwnd = 0    
    def enumHandler(self, hwnd, lParam):
        if win32gui.IsWindowVisible(hwnd):
            if 'Tiger64' in win32gui.GetWindowText(hwnd):
                self.hwnd = hwnd
    
    def __init__(self):
        win32gui.EnumWindows(self.enumHandler, None)


    def getRGB(self):
        hwnd = self.hwnd
        wDC = win32gui.GetWindowDC(hwnd)
        dcObj=win32ui.CreateDCFromHandle(wDC)
        cDC=dcObj.CreateCompatibleDC()
        dataBitMap = win32ui.CreateBitmap()
        rect = win32gui.GetWindowRect(hwnd)
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]
        dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
        cDC.SelectObject(dataBitMap)
        cDC.BitBlt((0,0),(w, h) , dcObj, (0,0), win32con.SRCCOPY)
        bits = np.array(dataBitMap.GetBitmapBits(), np.uint8)
        #dataBitMap.SaveBitmapFile(cDC, r'D:\bitmap.bmp')
        # Free Resources
        dcObj.DeleteDC()
        cDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, wDC)
        win32gui.DeleteObject(dataBitMap.GetHandle())    
        rgba = bits.reshape(h, w, 4).transpose()
        
        return rgba

if __name__ == '__main__':
    
    ts = TigerScreen()
    rgba = ts.getRGB()
    plt.imshow(rgba[0].transpose(), 'Blues')
    plt.show()
    plt.imshow(rgba[1].transpose(), 'Greens')
    plt.show()
    plt.imshow(rgba[2].transpose(), 'Reds')
    plt.show()