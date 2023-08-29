import numpy as np
import cv2
import struct #secret sauce   bytes->number->bytes
#bytes to thing :  UNPACK
#thing to bytes :  PACK
#https://docs.python.org/3/library/struct.html

img=cv2.imread("squirrel.jpg",0)
w,h=img.shape
print(img)
f= open("test2.ram","wb")
f.write(b"KADEN")
aSeriesOfBytes=struct.pack("<2I",w,h)
print(aSeriesOfBytes)
f.write(aSeriesOfBytes)
for row in img:
	for pixel in row:
		f.write(struct.pack("<B",pixel))
f.close()
	



