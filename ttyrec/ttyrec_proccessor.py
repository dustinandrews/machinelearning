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
import os

class ttyrec_proccessor:

    def __init__(self):
        self.data_shape = (24, 80)
        shape = (0, 24, 80)
        atom = tables.UInt8Atom()
        filters = tables.Filters(complevel=5, complib='zlib')
        self.h5f = tables.open_file('tty.h5', mode='w')
        if not 'earray' in self.h5f.root:
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
            #ensure the data is a standard shape regardless of TTY size.
            data = np.zeros((self.data_shape), dtype=np.int8)
            d1 = min(screen.shape[0], self.data_shape[0])
            d2 = min(screen.shape[1], self.data_shape[1])
            data[0:d1, 0:d2] += screen[0:24, 0:80] #copy the rectangle to 1,1            
            data = data[np.newaxis, :, :]            
            self.data.append(data)
        
    
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
        