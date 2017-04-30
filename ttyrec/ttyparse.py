#TTYRec handling 
from time import sleep
import pyte

#%%



def readHeader(recordingFile):
    sec = int.from_bytes(recordingFile.read(4), byteorder='little')
    usec = int.from_bytes(recordingFile.read(4), byteorder='little')
    length = int.from_bytes(recordingFile.read(4), byteorder='little')
    return(sec, usec, length)

#%%

screen = pyte.Screen(80,20)
ttyStream = pyte.ByteStream()
ttyStream.attach(screen)

rec_fileName = r"recordings\2008-03-25.14_36_24.ttyrec"
outfileName = r"recordings\2008-03-25.14_36_24.csv"
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
    x = input("#>")
    #if delay > 1:
    x = input("#")
    if x == "q":
        break
        

print("done")
print(recordingFile.tell())
