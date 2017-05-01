# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 16:11:45 2017

@author: dandrews
TTYRec annotation program

"""
import bz2
import os
import glob
import TtyParse


class Anotater:

    def decompress_file(self, filepath):
        newfilepath = filepath[:-4]
        if not os.path.exists(newfilepath):
            with open(newfilepath, 'wb') as new_file, bz2.BZ2File(filepath, 'rb') as file:
                 for data in iter(lambda : file.read(100 * 1024), b''):
                     new_file.write(data)
        return newfilepath
    
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
        

if __name__ == "__main__":
    
    self = Anotater()
    tty_files = glob.glob('./*/*/*.ttyrec.bz2')
    for file in tty_files:
        d = self.decompress_file(file)
        self.annote_recording(d)
        #os.remove(d)
        break
        