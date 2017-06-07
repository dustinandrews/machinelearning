# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 19:43:44 2017

@author: dandrews
"""

import time
import win32con, win32api, win32gui, atexit
from collections import namedtuple
import sys
import pyautogui

KeyboardEvent = namedtuple('KeyboardEvent', ['event_type', 'key_code',
                                             'scan_code', 'alt_pressed',
                                             'time'])

class keyboard_monitor():

    handlers = []
    keep_running = True
    last_update = time.time()
    
    def listen(self):
        """
        Calls `handlers` for each keyboard event received. This is a blocking call.
        """
        # Adapted from http://www.hackerthreads.org/Topic-42395
        from ctypes import windll, CFUNCTYPE, POINTER, c_int, c_void_p, byref
    
    
        event_types = {win32con.WM_KEYDOWN: 'key down',
                       win32con.WM_KEYUP: 'key up',
                       0x104: 'key down', # WM_SYSKEYDOWN, used for Alt key.
                       0x105: 'key up', # WM_SYSKEYUP, used for Alt key.                 
                      }
    
        def low_level_handler(nCode, wParam, lParam):
            """
            Processes a low level Windows keyboard event.
            """
            event = KeyboardEvent(event_types[wParam], lParam[0], lParam[1],
                                  lParam[2] == 32, lParam[3])
            for handler in self.handlers:
                handler(event)
    
            # Be a good neighbor and call the next hook.
            return windll.user32.CallNextHookEx(hook_id, nCode, wParam, lParam)
    
        # Our low level handler signature.
        CMPFUNC = CFUNCTYPE(c_int, c_int, c_int, POINTER(c_void_p))
        # Convert the Python handler into C pointer.
        pointer = CMPFUNC(low_level_handler)
    
        # Hook both key up and key down events for common keys (non-system).
        hook_id = windll.user32.SetWindowsHookExA(win32con.WH_KEYBOARD_LL, pointer,
                                                 win32api.GetModuleHandle(None), 0)
    
        # Register to remove the hook when the interpreter exits. Unfortunately a
        # try/finally block doesn't seem to work here.
        atexit.register(windll.user32.UnhookWindowsHookEx, hook_id)
    
        while self.keep_running:
            msg = win32gui.GetMessage(None, 0, 0)
            win32gui.TranslateMessage(byref(msg))
            win32gui.DispatchMessage(byref(msg))

    def print_event(self, e: KeyboardEvent):
        short_code = e.key_code & 0xFFFFFFFF        
        print(short_code, e.event_type, time.time())        

if __name__ == '__main__':
    
    kbm = keyboard_monitor()
    kbm.handlers.append(kbm.print_event)
    kbm.listen()
