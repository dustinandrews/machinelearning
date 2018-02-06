# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 17:00:26 2018

@author: dandrews

With help from the lmj nethack client https://github.com/lmjohns3/shrieker
"""

from pyte import Screen, ByteStream
import telnetlib
from sprites import SpriteSheet
import numpy as np
import matplotlib.pyplot as plt
from nhdata import NhData
import collections

class Point:
    __slots__ = ['x', 'y']
    def __init__(self, x, y):
        self.x = x
        self.y = y

MapXY = collections.namedtuple('mapxy', 'x y')

class NhClient:
    game_address = 'localhost'
    game_port = 23
    cols = 80
    rows = 35
    encoding = 'ascii'
    history = []
    data_history = []
    wb_message = b'welcome back to NetHack!'
    MAX_GLYPH = 1012
    map_x_y = MapXY(21,80)
    nhdata = NhData()
    cursor = Point(0,0)
    monster_count = len(nhdata.monsters.monster_data)
    tn = None
    _more_prompt = b'ore--\x1b[27m\x1b[3z'


    def __init__(self, username='aa'):
        self.username = username
        self.sprite_sheet = SpriteSheet("sprite_sheets/chozo32.bmp", 40, 30)
        self._init_screen()
        #self.start_session()

    def __del__(self):
         self.close()

    def start_session(self):
        self._init_screen()
        self.tn = telnetlib.Telnet(self.game_address)
        prompt = b'=>'
        self.tn.read_until(prompt,2)
        self.send_and_read_to_prompt(prompt, b'l')
        message = self.username.encode(self.encoding) + b'\n'
        self.send_and_read_to_prompt(prompt, message)
        self.send_and_read_to_prompt(prompt, message)
        self.send_and_read_to_prompt(prompt, b'p') # play

        page = self.history[-1]
        stale = False
        if self.is_stale(page):
            stale = True
            while stale:
                data = self.tn.read_until(b'seconds.', 1)
                page = self.render_data(data)
                self.history.append(page)
                stale = self.is_stale(page)
            self.tn.read_until(self._more_prompt, 2)


        while self.is_more(self.screen.display) or self.is_blank(self.screen.display):
            self.send_and_read_to_prompt(self._more_prompt, b'\n')
        #[print(line) for line in self.history]

    def reset_game(self):
        self.send_and_read_to_prompt(b'[yes/no]?', b'#quit\n')
        self.send_and_read_to_prompt(b'(end)', b'yes\n')
        page = self.history[-1]
        while self.is_end(page) or self.is_more(page):
            self.send_and_read_to_prompt(self._more_prompt, b' ')
            page = self.history[-1]
        self.close()


    def is_blank(self, page):
        for line in page:
            for c in line:
                if c != ' ':
                    return False
        return True

    def is_end(self, page):
        for line in page:
            if '(end)' in line:
                return True

    def is_stale(self, page):
        for line in page:
            if 'stale' in line:
                print(line)
                return True
        return False

    def is_more(self, page):
        for line in page:
            if '--More--' in line:
                return True
        return False

    def render_glyphs(self):
        """
        Creates a three channel numpy array and copies the correct glyphs
        to the array.

        Compatible with png and matplotlib
        ex:
            png.from_array(img.tolist(), 'RGB').save('map.png')
        """
        screen = np.zeros((self.map_x_y.x*32,self.map_x_y.y*32,3))
        glyphs = self.buffer_to_npdata()
        for row in range(len(glyphs)):
            for col in range(len(glyphs[row])):
                    glyph  = glyphs[row,col]
                    tile = self.sprite_sheet.get_image_by_number(int(glyph))
                    screen[row*32:(row*32)+32,col*32:(col*32)+32,:] = tile
        return screen

    def send_and_read_to_prompt(self, prompt, message, timeout=2):
        if type(prompt) == str:
            prompt = prompt.encode('ascii')

        if type(message) == str:
            message = message.encode('ascii')

        self.tn.write(message)
        print(prompt, message)
        data = self.tn.read_until(prompt, timeout)
        data += self.tn.read_very_eager()
        self.data_history.append(data)
        screen = self.render_data(data)
        #print(screen)
        self.history.append(screen)
        return data


    def close(self):
        self.send_string('S')
        self.send_string('y\n')
        print("closing")
        if self.tn:
           self.tn.close()

    def _init_screen(self):
        self.byte_stream = ByteStream()
        self.screen = Screen(self.cols,self.rows)
        self.byte_stream = ByteStream()
        self.byte_stream.attach(self.screen)

    def render_data(self, data):
        self.byte_stream.feed(data)
        lines = self.screen.display
        self.cursor.x = self.screen.cursor.x
        self.cursor.y = self.screen.cursor.y - 1 # last char is just before cursor
        return lines

    def buffer_to_npdata(self):
        skiplines = 1
        npdata = np.zeros((self.map_x_y.x, self.map_x_y.y), dtype=np.float32)
        npdata += 829 # set default to solid rock
        for line in range(skiplines,self.map_x_y.x+skiplines):
            for row in range(self.map_x_y.y):
                if self.screen.buffer[line] == {}:
                    continue
                glyph = self.screen.buffer[line][row].glyph
                if not self.screen.buffer[line][row].data == ' ':
                    npdata[line-skiplines,row] = glyph

        return npdata

    def buffer_to_rgb(self):
        npdata = self.buffer_to_npdata()
        min_m, max_m = self.nhdata.monsters.minkey, self.nhdata.monsters.maxkey
        min_o, max_o = self.nhdata.objects.minkey, self.nhdata.objects.maxkey
        min_r, max_r = self.nhdata.rooms.minkey, self.nhdata.rooms.maxkey
        r,b,g = npdata.copy(), npdata.copy(), npdata.copy()

        r = self._normalize_layer(r, min_m, max_m) # creatures
        b = self._normalize_layer(b, min_r, max_r) # room
        g = self._normalize_layer(g, min_o, max_o) # object

        rgb = np.zeros((r.shape + (3,)))
        rgb[:,:,0] = r
        rgb[:,:,1] = b
        rgb[:,:,2] = g

        return rgb

    def _normalize_layer(self, data, min_val, max_val):
        backup = data.copy()
        data[data < min_val] = min_val - 1
        data[data > max_val] = min_val - 1
        data += -(min_val - 1)
        data /= max_val

        if data.min() < 0:
            raise ValueError("dang")
        return data

    def imshow_map(self):
        img = self.render_glyphs()
        fig, ax = plt.subplots(figsize=(12,4))
        ax.axis('off')
        ax.imshow(img)
        plt.tight_layout()

    def get_visible_mobs(self):
        npdata = self.buffer_to_npdata()
        mobs = np.argwhere(npdata < self.monster_count)
        visible = []
        for mob in mobs:
            if not np.array_equal(mob, [self.cursor.x, self.cursor.y]):
                visible.append([mob, npdata[mob[0],mob[1]]])
        return visible

    def get_status(self):
        return self.nhdata.get_status(self.screen.display)

    def send_command(self, action_num):
        command = self.nhdata.COMMANDS[action_num]
        data = command.command
        self.send_and_read_to_prompt(b'\x1b[3z', data.encode('ascii'))

    def send_string(self, string):
        self.send_and_read_to_prompt(b'\x1b[3z', string)



if __name__ == '__main__':
    sampledata = b'\x1b[2;0z\x1b[2;1z\x1b[H\x1b[K\x1b[2;3z\x1b[2J\x1b[H\x1b[2;1z\x1b[2;3z\x1b[4;69H\x1b[0;832z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;833z-\x1b[1z\x1b[0m\x1b[5;69H\x1b[0;830z|\x1b[1z\x1b[0;848z\x1b[0m\x1b[1m\x1b[30m.\x1b[1z\x1b[0;16z\x1b[0m\x1b[1m\x1b[37m\x1b[7md\x1b[0m\x1b[0m\x1b[1z\x1b[0;848z\x1b[1m\x1b[30m.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;830z\x1b[0m|\x1b[1z\x1b[0m\x1b[6;69H\x1b[0;830z|\x1b[1z\x1b[0;45z\x1b[0m\x1b[1m\x1b[37mh\x1b[1z\x1b[0;848z\x1b[0m\x1b[1m\x1b[30m.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;830z\x1b[0m|\x1b[1z\x1b[0m\x1b[7;69H\x1b[0;844z\x1b[1m\x1b[31m+\x1b[1z\x1b[0;848z\x1b[0m\x1b[1m\x1b[30m.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;830z\x1b[0m|\x1b[1z\x1b[0m\x1b[8;69H\x1b[0;830z|\x1b[1z\x1b[0;848z\x1b[0m\x1b[1m\x1b[30m.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;830z\x1b[0m|\x1b[1z\x1b[0m\x1b[9;70H\x1b[0;848z\x1b[1m\x1b[30m.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;830z\x1b[0m|\x1b[1z\x1b[0m\x1b[10;69H\x1b[0;830z|\x1b[1z\x1b[0;848z\x1b[0m\x1b[1m\x1b[30m.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;848z.\x1b[1z\x1b[0;830z\x1b[0m|\x1b[1z\x1b[0m\x1b[11;69H\x1b[0;834z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;831z-\x1b[1z\x1b[0;835z-\x1b[1z\x1b[0m\x1b[6;70H\x1b[2;2z\x1b[23;1H\x1b[K[\x1b[7m\x08\x1b[1m\x1b[32m\x1b[CAa the Stripling\x1b[0m\x1b[0m\x1b[0m\r\x1b[23;18H]          St:18/02 Dx:14 Co:16 In:8 Wi:9 Ch:8  Lawful S:0\r\x1b[24;1HDlvl:1  $:0  HP:\x1b[K\r\x1b[1m\x1b[32m\x1b[24;17H18(18)\x1b[0m\r\x1b[24;23H Pw:\r\x1b[1m\x1b[32m\x1b[24;27H1(1)\x1b[0m\r\x1b[24;31H AC:6  Xp:1/0 T:1\x1b[2;1z\x1b[HVelkommen aa, the dwarven Valkyrie, welcome back to NetHack!\x1b[K\x1b[2;3z\x1b[6;70H\x1b[3z'

#%%

    nhc = NhClient()
    nhc.start_session()
    rgb = nhc.buffer_to_rgb()
    plt.imshow(rgb)


#    nh.byte_stream.feed(b''.join(nh.nhdata.SAMPLE_DATA))

#    nh.imshow_map()
#    import png
#    img = nh.render_glyphs()
#    png.from_array(img.tolist(), 'RGB').save('map.png')
#    npdata = nh.buffer_to_npdata()
