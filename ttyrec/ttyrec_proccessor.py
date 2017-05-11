# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 16:11:45 2017

@author: dandrews
TTYRec annotation program

"""
import bz2
import os
import glob
from ttyrec import ttyparse
import tables
from tqdm import tqdm
import numpy as np


class ttyrec_proccessor:

    def __init__(self):
        shape = (0, 24, 80)
        atom = tables.UInt8Atom()
        filters = tables.Filters(complevel=5, complib='zlib')
        self.h5f = tables.open_file('tty.h5', mode='w')
        self.data = self.h5f.create_earray(self.h5f.root, 'earray', atom, shape, filters=filters)
        
    
    def close(self):
        self.h5f.close()
        
        
    def decompress_file(self, filepath):
        newfilepath = filepath[:-4]
        if not os.path.exists(newfilepath):
            with open(newfilepath, 'wb') as new_file, bz2.BZ2File(filepath, 'rb') as file:
                 for data in iter(lambda : file.read(100 * 1024), b''):
                     new_file.write(data)
        return newfilepath
    
    def process_file(self, path):
        tty_file = self.decompress_file(path)
        parse = ttyparse.TtyParse(tty_file)
        for screen in tqdm(parse.frame_iterator(), total=parse.metadata.num_frames):
            self.data.append(screen[np.newaxis, :, :])
        
    
    def annote_recording(self, filename):
        parse = TtyParse.TtyParse(filename)
        metadata = parse.get_metadata()
        summaryfilename = filename[:-6] + "txt"
        start = len(metadata.frames) -10
        end = len(metadata.frames)
        parse.render_frames(start, end)
        
        with open(summaryfilename, "w") as sum_f:
            sum_f.write("   Start: {}\n     End: {}\nDuration: {}\n F count: {}\n\n".format(
                metadata.start_time,
                metadata.end_time,
                metadata.duration,
                len(metadata.frames)))
            sum_f.write("\n".join(parse.screen.display))
            
        print(summaryfilename)
        

#if __name__ == "__main__":
#    
#    self = Anotater()
#    tty_files = glob.glob('./*/*/*.ttyrec.bz2')
#    for file in tty_files:
#        d = self.decompress_file(file)
#        self.annote_recording(d)
#        #os.remove(d)
#        break
        