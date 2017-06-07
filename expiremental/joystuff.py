# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 10:22:44 2017

@author: dandrews
"""

import pygame
import time
import numpy as np

class joystickInspector:

    def __init__(self):
        pygame.display.init()
        pygame.joystick.init()
        joysticks = [pygame.joystick.Joystick(x) for x in range(pygame.joystick.get_count())]
        joy0 = joysticks[0]
        joy0.init()
        self.axis_count = joy0.get_numaxes()
        self.button_count = joy0.get_numbuttons()
        self.hat_count = joy0.get_numhats()
        self.joy = joy0
        self.total_inputs = self.axis_count  + self.button_count + (2 * self.hat_count) 
        
    def getJoyStickData(self):
        running = 1
        data = [0] * self.total_inputs
        pygame.event.pump()
        for i in range(self.axis_count):
            #print("{:8.2f} ".format(self.joy.get_axis(i)),end="",)
            data[i] = self.joy.get_axis(i)
        #print("   ", end="")
        running += i
        for i in range(self.button_count):
            #print(self.joy.get_button(i), end="")
            data[i+running] = self.joy.get_button(i)
        #print("   ", end="")
        running += i
        for i in range(self.hat_count):
            #print(self.joy.get_hat(i), end ="")
            a,b = self.joy.get_hat(i)
            data[i+running] = a
            i += 1
            data[i+running] = b
        return data
            
if __name__ == "__main__":
    
    j = joystickInspector()
    print( j.total_inputs )
    while True:
        print(j.getJoyStickData())
        time.sleep(0.5)
        
    
    
    
    

