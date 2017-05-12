# -*- coding: utf-8 -*-
"""
Created on Thu May 11 09:55:55 2017

@author: dandrews
"""

import glob
import colorama
from tqdm import tqdm

#from ttyrec import ttyparse
from ttyrec.ttyrec_proccessor import ttyrec_proccessor


file_names = glob.glob('./ttyrec/recordings/*/*.bz2')

proc = ttyrec_proccessor()
for fname in file_names[5:10]:
    try:
        proc.process_file(fname)
    except:
        proc.close()
        raise
proc.close()
    
    

#self = ttyparse.TtyParse(glob.glob('./ttyrec/recordings/*/*.ttyrec')[0])
#meta_data =self.get_metadata()
#print("   Start: {}\n     End: {}\nDuration: {}\n Frame count: {}".format(
#        meta_data.start_time,
#      meta_data.end_time,
#      meta_data.duration,
#      meta_data.num_frames))
#print("   lines: {}\n columns: {}".format(meta_data.lines, meta_data.collumns))
#
#datafile = self.rec_filename.replace("ttyrec", "hf5")
#
#for screen in tqdm(self.frame_iterator(), total=self.metadata.num_frames):
    
#        sc = self.get_next_render()

    
#%% 
    
#    root = hfileh.root    
#    group = hfileh.create_group(root, "renders")
#    table = hfileh.create_table("/renders", "renders", tty_record, "Render lines")
#    row = table.row
    
#    shape = (self.metadata.num_frames, self.metadata.lines, self.metadata.collumns)
#    atom = tables.UInt8Atom()
#    filters = tables.Filters(complevel=5, complib='zlib')
#    h5f = tables.open_file('tty.h5', mode='w')
#    data = h5f.create_carray(h5f.root, 'carray', atom, shape, filters=filters)
#    
#    
#
#    for i in tqdm(range(self.metadata.num_frames)):
#        data[i] = self.get_next_frame_as_np()
#    
#    h5f.close()
#%%


    #hfileh = tables.open_file('tty.h5', mode='r')