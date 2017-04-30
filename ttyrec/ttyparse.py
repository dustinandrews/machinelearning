#TTYRec handling 
from time import sleep
from pyte import Screen, ByteStream
#import colorama
import datetime
from TestScreen import TestScreen
cursor_home = '\033[1;1H'


class Framedata:
    sec = 0
    usec = 0
    length = 0
    start_pos = 0
    
    def __init__(self, sec, usec, length, start_pos):
        self.sec, self.usec, self.length, self.start_pos = sec, usec, length, start_pos
 
class Metadata:
    start_time = 0
    end_time = 0
    duration = None
    frames = None

class Ttyparse():    
    metadata = None
    def __init__(self, rec_filename):
        self.rec_filename = rec_filename
        
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
    
    def get_metadata(self):
        
        # don't reprocess, but allow multipe calls
        if self.metadata != None:
            return self.metadata
        
        framedata = []
        with open(self.rec_filename, "rb") as ttyrec_file:        
            while True:
                sec, usec, length = self.read_header(ttyrec_file)
                if length == 0:
                    break
                frame = Framedata(sec = sec, usec=usec, length=length, start_pos=ttyrec_file.tell())
                ttyrec_file.read(length)
                #frame = {'sec': sec, 'usec': usec, 'length': length, 'start_pos': ttyrec_file.tell()}
                framedata.append(frame)

        metadata = Metadata()
        metadata.start_time = framedata[0].sec + framedata[0].usec / 1e6
        metadata.end_time = framedata[-1].sec + framedata[-1].usec / 1e6
        metadata.duration = datetime.timedelta(seconds=metadata.end_time - metadata.start_time)
        metadata.frames = framedata
        self.metadata = metadata
        return metadata


    """
    returns byte data. ASCII or UTF8, unkown. data.decode("utf-8", 'ignore') to get text 
    """
    def get_frame(self, frame_data: Framedata):
        with open(self.rec_filename, "rb") as ttyrec_file:
            ttyrec_file.seek(frame_data.start_pos)
            data = ttyrec_file.read(frame_data.length)
            return data


    def detect_screen_size(self):
        self.get_metadata()
    
                
if __name__ == "__main__":
    import glob
    
    self = Ttyparse(glob.glob('./*/*/*.ttyrec')[0])
    meta_data =self.get_metadata()
    print("   Start: {}\n     End: {}\nDuration: {}\n F count: {}".format(meta_data.start_time,meta_data.end_time,meta_data.duration,len(meta_data.frames)))
#    for i in range(-50, -1):
#        print(self.get_frame(meta_data.frames[i]).decode('utf8', 'ignore'))
#        

    byte_stream = ByteStream()
    screen = TestScreen(200,100)
    byte_stream.attach(screen)
    for fdata in meta_data.frames[:20]:
        byte_stream.consume(self.get_frame(fdata))
    
    for line in screen.display:
            print(line, end="")
    print(screen.maxline, screen.maxcolumn)    
    
        