#TTYRec handling 
from time import sleep
from pyte import Screen, ByteStream
#import colorama
import datetime
from .TestScreen import TestScreen
import os
import glob
#import colorama
import colorama
colorama.init(strip=False)
from tqdm import tqdm


class Framedata:
    sec = 0
    usec = 0
    length = 0
    start_pos = 0
    timestamp = 0
    
    def __init__(self, sec, usec, length, start_pos):
        self.sec, self.usec, self.length, self.start_pos = sec, usec, length, start_pos
        self.timestamp = self.sec + self.usec / 1e6
 
class Metadata:
    start_time = 0
    end_time = 0
    duration = None
    frames = None
    lines = 0
    collumns = 0
    num_frames = 0

class TtyParse():
    cursor_home = '\033[1;1H'
    clear_screen = '\033[2J'  
    metadata = None
    rendered_frames = []
    
    def __init__(self, rec_filename):        
        self.rec_filename = rec_filename
        self.last_rendered = 1
        self.get_metadata()
        
        
    """
    TTYRec
    sec = seconds
    usec = decimal seconds
    length = length of next data segment 
    """
    def read_header(self, file_handle):
        sec = int.from_bytes(file_handle.read(4), byteorder='little')
        usec = int.from_bytes(file_handle.read(4), byteorder='little')
        length = int.from_bytes(file_handle.read(4), byteorder='little')
        return(sec, usec, length)
    
    def get_metadata(self, scan_limit = 100):
        print("Parsing file...")
        byte_stream = ByteStream()
        screen = TestScreen(200,100)
        byte_stream.attach(screen)        
        
        # don't reprocess, but allow multipe calls
        if self.metadata != None:
            return self.metadata
        
        framedata = []
        index = 0
        filesize = os.path.getsize(self.rec_filename)
        with tqdm(total=filesize, ascii=False, desc="bytes") as progress_bar:
            with open(self.rec_filename, "rb") as ttyrec_file:        
                while True:
                    sec, usec, length = self.read_header(ttyrec_file)
                    progress_bar.update(length + 12)
                    if length == 0:
                        break
                    frame = Framedata(sec = sec, usec=usec, length=length, start_pos=ttyrec_file.tell())
                    data = ttyrec_file.read(length)
                    if(index < scan_limit): # scan frames up to the limit
                        byte_stream.consume(data)
                    index += 1
                    #frame = {'sec': sec, 'usec': usec, 'length': length, 'start_pos': ttyrec_file.tell()}
                    framedata.append(frame)
                    

        print()
        metadata = Metadata()
        metadata.start_time = framedata[0].sec + framedata[0].usec / 1e6
        metadata.end_time = framedata[-1].sec + framedata[-1].usec / 1e6
        metadata.duration = datetime.timedelta(seconds=metadata.end_time - metadata.start_time)
        metadata.frames = framedata
        metadata.num_frames = len(framedata)
        metadata.lines = screen.maxline + 1
        metadata.collumns = screen.maxcolumn + 1
        self.metadata = metadata
        self.screen = screen
        self.init_screen()
        return metadata


    """
    returns byte data. UTF8 sorta. data.decode("utf-8", 'ignore') to get text
    Better yet use pyrec to consume the stream.
    """
    def get_frame(self, frame_data: Framedata):
        with open(self.rec_filename, "rb") as ttyrec_file:
            ttyrec_file.seek(frame_data.start_pos)
            data = ttyrec_file.read(frame_data.length)
            return data
    
    """
    Get a pyte screen for the recording
    """
    def init_screen(self):
        self.byte_stream = ByteStream()
        self.screen.resize(self.metadata.lines, self.metadata.collumns)
        self.byte_stream.attach(self.screen)
        
    """
    Runs the frames into the screen from start to end
    (-10,None) will do last 10 frames
    """
    def render_frames(self, start, end):
        for f in range(start,end):
            frame = self.get_frame(self.metadata.frames[f])           
            self.byte_stream.consume(frame)
            
    def get_next_render(self):
        if self.last_rendered == self.metadata.num_frames:
            self.last_rendered = 1
        self.render_frames(self.last_rendered - 1, self.last_rendered)
        lines = self.screen.display
        self.last_rendered += 1
        return lines
    
    """
    Gets the next rendered frame as a flat array
    of acii values (-32 to make space 0)
    """
    def get_next_render_flat(self):
        x = self.get_next_render()
        return [ord(item)-32 for l in x for item in l]
        
if __name__ == "__main__":
   # import glob
    import pickle
    
    self = TtyParse(glob.glob('./*/*/*.ttyrec')[0])
    meta_data =self.get_metadata()
    print("   Start: {}\n     End: {}\nDuration: {}\n Frame count: {}".format(
            meta_data.start_time,
          meta_data.end_time,
          meta_data.duration,
          meta_data.num_frames))
    print("   lines: {}\n columns: {}".format(meta_data.lines, meta_data.collumns))

    datafile = self.rec_filename.replace("ttyrec", "p")
    
    this_might_get_big = []
    
    m = 0
    
    for i in tqdm(range(meta_data.num_frames)):
        sc = self.get_next_render_flat()
        for i in sc:
            if i > m:
                m = i
    print(m)
            
            
    
   
