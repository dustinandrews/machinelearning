#TTYRec handling 
from time import sleep
import pyte
import colorama
cursor_home = '\033[1;1H'

#%%



def readHeader(recordingFile):
    sec = int.from_bytes(recordingFile.read(4), byteorder='little')
    usec = int.from_bytes(recordingFile.read(4), byteorder='little')
    length = int.from_bytes(recordingFile.read(4), byteorder='little')
    return(sec, usec, length)

#%%

screen = pyte.Screen(150,50)
ttyStream = pyte.ByteStream()
ttyStream.attach(screen)

rec_fileName = r"recordings\stth\2015-08-16.06_56_37.ttyrec"
outfileName = r"recordings\stth\2015-08-16.06_56_37.csv"
recordingFile = open(rec_fileName, "rb")
outfile = open(outfileName, "w")
startTime = 0.0
cont = True
index = 0
while cont == True:
    headers = readHeader(recordingFile)
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
        noop = 0
    startTime = thisTime
    if headers[2] == 0:
        cont = False
    else:
        payload = recordingFile.read(headers[2])    
    #pString = payload.decode("utf-8")
    #pString = re.sub(u'(\002?\033\[\?\d+[a-zA-Z])', u'', pString)
    #print(pString, end="")
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
        

print("done")
print(recordingFile.tell())
