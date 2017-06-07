# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 16:32:43 2017

@author: dandrews
"""

import sys
if not r'D:\local\vXboxInterface-x64' in sys.path:
    sys.path.append(r'D:\local\vXboxInterface-x64')
from vxboxinterfacepy import xbox_vcontrol
from game_recorder import windows_screen_grab
import keras
import numpy as np

model_name = 'trained.model'
model = keras.models.load_model(model_name)
#xb = xbox_vcontrol(2)


class input_container:
    lx = 0
    ly = 0
    rx = 0
    ry = 0
    xb_cont = xbox_vcontrol(2)
    max_input = 32768
    
    def __init__(self):
        self.xb_cont.plug_in()
    
    
    def map_prediction(self, prediction):
        #print()
      
        lx = int(prediction[0] * self.max_input * -1)
        ly = int(prediction[1] * self.max_input)
        #skip [1], it's the triggers
        rx = int(prediction[4] * self.max_input)
        ry = int(prediction[3] * self.max_input)
        #print(lx, ly, rx, ry)
        self.xb_cont.set_axis_lx(lx)
        self.xb_cont.set_axis_ly(ly)
        self.xb_cont.set_axis_rx(rx)
        self.xb_cont.set_axis_ry(ry)
        return (lx, ly, rx, ry)

#%%
if __name__ == '__main__':
    
    
    ic = input_container()
    
    import tables
    from matplotlib import pyplot as plt
    test_data = tables.open_file('t-processed.h5','r')
    plt.imshow(test_data.root.data[0][0])
    saved_data = test_data.root.data
    saved_labels = test_data.root.labels

    
#%%   
    errors = np.zeros(len(saved_data)) 
    for index in range(len(saved_data)):
        prediction = model.predict([saved_data[index].reshape((1,)+ saved_data[0].shape)])[0]
        mse = np.mean((prediction - saved_labels[index]) **2)
        errors[index] = mse
    plt.plot(mse)
    
    

#%%    
#    while True:
#        data = data_cycle()
#        prediction = model.predict([data])[0]
#        print(ic.map_prediction(prediction) )
#        break