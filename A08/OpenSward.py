import numpy as np
import cv2
import struct #secret sauce   bytes->number->bytes
#bytes to thing :  UNPACK
#thing to bytes :  PACK
#https://docs.python.org/3/library/struct.html

f= open("test.ram","rb")
x=f.read(5)
if x!=b"KADEN":
	print("invalid file")
w,h=struct.unpack("<2I",f.read(8))
print(w,h)
pixels=[]
while 1:
	b=f.read(1)
	if not b:
		break
	pixels.append(struct.unpack("<B",b))
img=np.reshape(pixels,(w,h))
cv2.imwrite("test.png",img)
print(img)


	#return throw exception ...
# ~ aSeriesOfBytes=struct.pack("<2I",w,h)
# ~ print(aSeriesOfBytes)
# ~ f.write(aSeriesOfBytes)
# ~ for row in img:
	# ~ for pixel in row:
		# ~ f.write(struct.pack("<B",pixel))

# ~ f.close()
	



