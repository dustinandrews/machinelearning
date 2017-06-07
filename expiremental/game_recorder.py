# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 12:59:24 2017

@author: dandrews
"""

import win32ui
import win32gui
import win32con
import numpy as np
import pygame
import time
import tables
import argparse
import tqdm
import colorama
colorama.init(convert=True)
from matplotlib import pyplot as plt
#%%


"""
Win32iu based screen grabbing tools. Allows capturing more screens per second
on windows than other python based tools. YMMV based on machine specs.
"""
class windows_screen_grab:
    hwnd = 0
    scale = 0.5 # TODO: Make this a parameter
    
    """
    Define a callback that finds the correct window handle
    """
    def enumHandler(self, hwnd, lParam):
        if win32gui.IsWindowVisible(hwnd):
            if self.search_str in win32gui.GetWindowText(hwnd):
                self.hwnd = hwnd
    
    """
    window_title: a whole or partial case sensitive search string for the
                  window to attach and capture.
    """
    def __init__(self, window_title: str):
        self.search_str = window_title
        win32gui.EnumWindows(self.enumHandler, None)
        hwnd = self.hwnd
        rect = win32gui.GetWindowRect(hwnd)
        w = rect[2] - rect[0]
        h = rect[3] - rect[1]
        dest_w = int(w * self.scale)
        dest_h = int(h * self.scale)
        dataBitMap = win32ui.CreateBitmap()        
        w_handle_DC = win32gui.GetWindowDC(hwnd)
        windowDC = win32ui.CreateDCFromHandle(w_handle_DC)
        memDC = windowDC.CreateCompatibleDC()
        dataBitMap.CreateCompatibleBitmap(windowDC , dest_w, dest_h)
        memDC.SelectObject(dataBitMap)
        self.dataBitMap = dataBitMap
        self.memDC = memDC
        self.windowDC = windowDC
        self.h = h
        self.w = w
        self.dest_w = dest_w
        self.dest_h = dest_h
        self.rgb = np.zeros((3,dest_h, dest_w))
    

    """
    Get the raw screen bits.
    
    returns: a tuple of the bits in the format (r, g, b, a, r, g, b, a, ...)
    """
    def get_bits(self):        
        self.memDC.StretchBlt((0,0), (self.dest_w, self.dest_h), self.windowDC, (0,0), (self.w,self.h), win32con.SRCCOPY)        
        bits = np.fromstring(self.dataBitMap.GetBitmapBits(True), np.uint8)
        return bits
    
    
    """
    Strip the alpha channel and orient the images to match the original screen
    orientation. A model won't care if it's sideways, but humans do.
    """
    def get_rgb_from_bits(self, bits):        
        self.rgb[0] = bits[2::4].reshape(self.dest_h, self.dest_w)
        self.rgb[1] = bits[1::4].reshape(self.dest_h, self.dest_w)
        self.rgb[2] = bits[0::4].reshape(self.dest_h, self.dest_w)
        return self.rgb
    
    """
    Release resources. Call if you want to re-use your python kernel.
    """
    def cleanup(self):
        # Free Resources
        self.dcObj.DeleteDC()
        self.cDC.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, self.wDC)
        win32gui.DeleteObject(self.dataBitMap.GetHandle())    
    
#%%
"""
Use pygame to monitor the gamepad
"""
class input_grab:

    """
    Assumes you have just one gamepad.
    """
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
        
    def get_input_data(self):
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
#%% 
if __name__ == "__main__":
    

#%%
    """
    Write one sample to the data array
    """
    def getSample():
        start_time = time.time()        
        screen_data = screen.get_bits()
        control_data = joy.get_input_data()
        write_dict = {'data': screen_data, 'labels': control_data}        
        data.append(write_dict) # ~35.9 ms
        end_time = time.time()
        duration = end_time - start_time
        wait_time = min_time_per_loop - duration       
        time.sleep(np.max((0, wait_time)))
        end_time = time.time()
        duration = end_time - start_time
        

    """
    Write the data out to an hf5 file
    """
    def write_data(filename, data):
        data_file = tables.open_file(filename, 'w')
        s_atom = tables.Atom.from_dtype(np.dtype(np.uint8))
        screen_col = data_file.create_earray(
                data_file.root, 
                'data', 
                s_atom, 
                (0,) + (len(screen_data),))
        
        i_atom = tables.Atom.from_dtype(np.dtype(np.float32))
        input_col = data_file.create_earray(
                data_file.root, 
                'labels', 
                i_atom,(0,) +  (len(control_data),))
        
        for i in tqdm.trange(len(data)):
            d = data[i]
            screen_col.append([d['data']])
            input_col.append([d['labels']])
        data_file.close()
#%%        

    """
    Convert a data file with the raw uint8 data in flat format to a 3d array
    of np.float32 that is better suited to Keras Conv2D layers.
    """
    def process_raw_pixels(infile, outfile):
        try:
            h_in = tables.open_file(infile, 'r')
            h_out = tables.open_file(outfile, 'w')
            sample_data = screen.get_rgb_from_bits(h_in.root.data[0])
            sample_label = h_in.root.labels[0]
            
            f_atom = tables.Atom.from_dtype(np.dtype(np.float32))
            data_col = h_out.create_earray(
                    h_out.root,
                    'data',
                    f_atom,
                    (0,) + sample_data.shape
                    ) 
            label_col = h_out.create_earray(
                    h_out.root,
                    'labels',
                    f_atom,
                    (0,) + sample_label.shape
                    )
            
            for i in tqdm.trange(h_in.root.labels.nrows):
                label_col.append([h_in.root.labels[i]])
                new_data = screen.get_rgb_from_bits(h_in.root.data[i])
                data_col.append([new_data])
        except:
            raise
        finally:
            
            h_in.close()
            h_out.close()
#%%            
    """
    Record data for the number of specified seconds.
    """        
    def record_data(duration_secs):        
        end_time = time.time() + duration_secs
        index = 0
        while time.time() < end_time:
            getSample()
            index += 1
            if index % 10 == 0:
                print(".", end="")
            if index % 100 == 0:
                print(" {} frames.".format(index))
        print()
                
#%%
    parser = argparse.ArgumentParser(
            description="Capture screen and controller data",
            epilog="processed_file "
            )
    parser.add_argument('--window_search_str', 
                        help='Cases sensitive window search substring',
                        default='Game')
    parser.add_argument('--seconds',
                        type=int, 
                        help='number of seconds of input to capture',
                        default=10
                        )
    parser.add_argument('--outfile', 
                        help='name of h5 output file for raw recording',
                        default='arg_not_supplied',
                        )
    parser.add_argument('--processed_file', 
                        help='if supplied, just post processed --outfile to'+\
                        ' a 3d, float32 format. Other args ignored.',
                        default='arg_not_supplied')
    parser.add_argument('--samples_sec',
                        type=int,
                        help="samples to attempt per second. default = 20",
                        default=20)
    
    args = parser.parse_args()
#%%    
    """
    Create grabber classes
    """
    screen = windows_screen_grab(args.window_search_str)
    joy = input_grab()
    
    """
    Grab some sample data
    """
    screen_data = screen.get_bits()
    control_data = joy.get_input_data()    
   
    """
    All the data is held in memory in order to be fast enough to get several
    frames per second.
    """
    data = []
    
    """
    Limit the number of frames per second to grab and space them evenly over
    the second.    
    """
    min_time_per_loop = 1/args.samples_sec # 1/20 = 20 times/sec
    
    
    if args.outfile == 'arg_not_supplied':
        parser.print_help()
    else:    #parser.exit(1)        
        if args.processed_file == 'arg_not_supplied':
            print('Starting recording for {} seconds.'.format(args.seconds))
            record_data(args.seconds)
            print('writing data to {}, please stand by'.format(args.outfile))
            write_data(args.outfile, data)
        else:
            print('Converting {} to 3d style data as {}'.format(args.outfile,args.processed_file))
            process_raw_pixels(args.outfile,args.processed_file) 
            
#%%            
    def show_rgb():
        rgb = screen.get_rgb_from_bits(screen.get_bits())
        plt.imshow(rgb[0])
        plt.show()
        plt.imshow(rgb[1])
        plt.show()
        plt.imshow(rgb[2])
        plt.show()