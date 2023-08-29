import numpy as np
import cv2
import struct #secret sauce   bytes->number->bytes
#bytes to thing :  UNPACK
#thing to bytes :  PACK
#https://docs.python.org/3/library/struct.html
def show(img,wait=0,destroy=True):
	if img.min()<0 or img.max()>255:
		img=normalize(img)
	img=np.uint8(img)
	cv2.imshow("image",img)
	cv2.waitKey(wait)
	if destroy:
		cv2.destroyAllWindows()

def normalize(img):
	out=img*1.0
	out-=out.min()
	out/=out.max()
	out*=255.99
	return np.uint8(out)

def save(img,filename):
	w,h=img.shape
	f= open(filename+".ram","wb")
	f.write(b"RAM")
	aSeriesOfBytes=struct.pack("<2I",w,h)
	f.write(aSeriesOfBytes)
	pixels=""
	for row in img:
		for pixel in row:
			pixels+="1" if pixel>128 else "0"
	padSize=(8-len(pixels))%8
	pixels+="0"*padSize
	counter=0
	previous="0"
	opt=""
	for char in range(len(pixels)):
		if pixels[char]=="1" or pixels[char]=="0":
			if previous==pixels[char]:
				counter+=1
			else:
				if pixels[char]== "0":
					opt+=str(counter)+"!"
				elif pixels[char]=="1":
					opt+=str(counter)+"@"
				counter=1
				previous=pixels[char]
	if pixels[-1]=="1":
		opt+=str(counter)+"!"
	if pixels[-1]=="0":
		opt+=str(counter)+"@"
	f.write(bytes(opt,'utf-8'))
	f.close()

def read(filename):
	f= open("%s.ram"%filename,"rb")
	x=f.read(3)
	if x!=b"RAM":
		print("invalid file")
	w,h=struct.unpack("<2I",f.read(8))
	
	s=""
	while 1:
		b=f.read(1)
		if not b:
			break
		# ~ print(b)
		s+=str(b)
	# ~ print(s)
	s=s.replace("'", "")
	s=s.replace("b", "")
	counter=""
	previous=""
	decoded=""
	print(s)
	for char in range(len(s)):
		if s[char]=="@":
			decoded+= "0"*int(float(counter))
			counter=""
		elif s[char]=="!":
			decoded+= "1"*int(float(counter))
			counter=""
		else:
			counter+=s[char]
	print(decoded)
	padSize=(8-len(decoded))%8
	decoded+= "0"*padSize
	print(len(decoded))
	f.close
	f= open("test1.ram","wb")
	for i in range(0,len(decoded),8):
			x=int(decoded[i:i+8],2)
			f.write(struct.pack("<B",x))
	# ~ f.write(bytes(decoded, 'utf-8'))
	f.close
	f= open("test1.ram","rb")
	while 1:
		b=f.read(1)
		if not b:
				break
		v=struct.unpack("<B",b)[0]
		s+=bin(v)[2:].zfill(8)
	# ~ print(s)
	decoded+="0"*300
	m=np.array(list(decoded))
	img=np.uint8(np.reshape(m[:w*h],(w,h))=="1")*255
	show(img)
	return img

	# ~ decoded=decoded.replace("1","255")
	# ~ padSize=(8-len(decoded))%8
	# ~ print((8-len(decoded))%8)
	# ~ decoded+="0"*padSize
	# ~ doceded=int(decoded)
	# ~ m=np.array(list(decoded))
	# ~ print(m)
	# ~ img=np.uint8(np.reshape(m[:w*h],(w,h))=="1")*255
	# ~ cv2.imwrite("test.png",img)
	# ~ return img


img=cv2.imread("haiku_proof.bmp",0)
save(img,"test2")
cv2.imwrite("out2.png",read("test2"))
