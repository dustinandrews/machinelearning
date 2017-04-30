#TTYRec handling 
from time import sleep
import pyte
import colorama
import datetime
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
        return metadata


    """
    returns byte data. ASCII or UTF8, unkown. data.decode("utf-8", 'ignore') to get text 
    """
    def get_frame(self, frame_data: Framedata):
        with open(self.rec_filename, "rb") as ttyrec_file:
            ttyrec_file.seek(frame_data.start_pos)
            data = ttyrec_file.read(frame_data.length)
            return data
        
    def process_recording(self):    
        screen = pyte.Screen(150,50)
        ttyStream = pyte.ByteStream()
        ttyStream.attach(screen)
        
        #rec_fileName = r"recordings\stth\2015-08-16.06_56_37.ttyrec"
        outfileName = self.rec_filename.replace('.ttyrec', '.csv')
        
        with open(outfileName, "w") as outfile:
            startTime = 0.0
            cont = True
            index = 0
            while cont == True:
                headers = read_header(self.ttyrec_file)
                #print(headers)
                
                if startTime == 0:
                    startTime = (headers[0] + headers[1]/100) / 1000
                thisTime = (headers[0] + headers[1]/100) / 1000
                #print(thisTime, startTime)
                delay = thisTime - startTime
                if delay < 0:
                    delay = delay * -1 #WTF negative times?
                #print("delay", delay)
                startTime = thisTime
                if(delay < 100):
                    sleep(delay/50)
                    #noop = 0
                startTime = thisTime
                if headers[2] == 0:
                    cont = False
                else:
                    payload = self.ttyrec_file.read(headers[2])    
    
    
                ttyStream.feed(payload)
                for line in screen.display:
                    outfile.write("{}, ".format(index))
                    for c in line:
                        outfile.write( "{}, ".format(ord(c)))
            #%%
                for l in screen.display:
                    print(l)
                    
                index += 1
                x = input("#")
                if x == "q":
                    break
                print(cursor_home)
                
if __name__ == "__main__":
    import glob
    
    self = Ttyparse(glob.glob('./*/*/*.ttyrec')[0])
    meta_data =self.get_metadata()
    print("   Start: {}\n     End: {}\nDuration: {}\n F count: {}".format(meta_data.start_time,meta_data.end_time,meta_data.duration,len(meta_data.frames)))
    for i in range(-50, -1):
        print(self.get_frame(meta_data.frames[i]).decode('utf8', 'ignore'))
        