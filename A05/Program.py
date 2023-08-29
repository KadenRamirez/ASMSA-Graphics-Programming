import numpy as np
import cv2
import time
import math

DEBUG=True

class SeamCarveImage:
	#self is always the first argument or whatever goes beofre the dot.
	def __init__(self,filename,attractMask=None,repulseMask=None):
		self.grey=cv2.imread(filename,0)
		self.img=cv2.imread(filename)
		self.img=cv2.resize(self.img,(0,0),fx=.7,fy=.7)
		self.grey=cv2.resize(self.grey,(0,0),fx=.7,fy=.7)
		self.seamImg=self.img*1
		self.h,self.w=self.img.shape[:2]
		print(self.h)
		print(self.w)
	def removeVerticleSeam(self):
		seam=getSeam(self.grey)
		self.seamImg=drawSeam(self.seamImg,seam)
		# ~ show(self.seamImg,wait=True)
		self.grey=removeSeam(self.grey,seam)
		self.img=removeSeam(self.img,seam)
		cv2.imwrite("output/retargeter.jpg",np.uint8(self.img))
		
	def removeHorizontalSeam(self):
		self.grey=np.swapaxes(self.grey,0,1)
		self.img=np.swapaxes(self.img,0,1)
		self.seamImg=np.swapaxes(self.seamImg,0,1)
		seam=getSeam(self.grey)
		self.grey=removeSeam(self.grey,seam)
		self.img=removeSeam(self.img,seam)
		self.seamImg=drawSeam(self.seamImg,seam)
		self.grey=np.swapaxes(self.grey,0,1)
		self.img=np.swapaxes(self.img,0,1)
		self.seamImg=np.swapaxes(self.seamImg,0,1)
		# ~ show(self.seamImg, wait=True)
		
	def show(self,wait=False):
		show(self.img,wait=wait)

def show(img,title="image",wait=True):
    d=max(img.shape[:2])
    if d>1000:
        step=int(math.ceil(d/1000))
        img=img[::step,::step]
    if not DEBUG:
        return
    if np.all(0<=img) and np.all(img<256):
        cv2.imshow(title,np.uint8(img))
    else:
        print("normalized version")
        cv2.imshow(title,normalize(img))
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.waitKey(1)

def normalize(img):
    img_copy=img*1.0
    img_copy-=np.min(img_copy)
    img_copy/=np.max(img_copy)
    img_copy*=255.9999
    return np.uint8(img_copy)
    
def drawSeam(img,seam,color=255):
    for x,y in seam:
        img[y,x]=color
    return img

def getEdgeImage(img,margin=10):
    kernel=np.float64([[-1,0,1]])
    Ix=cv2.filter2D(img,cv2.CV_64F,kernel)
    Iy=cv2.filter2D(img,cv2.CV_64F,kernel)
    I=np.hypot(Ix,Iy)
    m=I.max()
    I[:,:margin]=m
    I[:,-margin:]=m
    return I
    
def getEnergyMap(img):
    edges=getEdgeImage(img)
    kernel=np.ones(3,np.float64)
    for i in range(1,len(edges)):
        minAbove=cv2.erode(edges[i-1],kernel).T[0]
        edges[i]+=minAbove
    return edges


def getSeam(img):
    energyMap=getEnergyMap(img)
    y=len(energyMap)-1
    x=np.argmin(energyMap[y])
    seam=[(x,y)]
    while len(seam)<len(energyMap):
        x,y=seam[-1]
        newY=y-1
        newX=x+np.argmin(energyMap[newY,x-1:x+2])-1
        seam.append((newX,newY))
    return seam

def removeSeam(img,seam):
    
    output=img[:,1:]
    for (x,y),row in zip(seam,img):
        output[y,:x]=img[y,:x]
        output[y,x:]=img[y,x+1:]
    return output

def reTarget(img,w=824,h=1120,tw=240,th=320):
	while h!=th or w!=tw:
		if h!=th:
			img.removeHorizontalSeam()
			img.show()
			h-=1
		elif w!=tw:
			img.removeVerticleSeam()
			img.show()
			w-=1
		elif w==tw and h==th:
			print(w)
			print(h)
			# ~ cv2.imwrite("output/retarget.jpg")
			# ~ img.show(wait=True)


img=SeamCarveImage("input/image1.jpg")
reTarget(img)

# ~ img1=cv2.imread("input/image1.jpg",0)
# ~ show(getEnergyMap(img1))
# ~ cv2.imwrite("output/EnergyMap.jpg", getEnergyMap(img1))

# ~ while 1:
	# ~ img.removeVerticleSeam()
	# ~ img.removeHorizontalSeam()
	# ~ img.show()


# ~ img1=cv2.imread("input/image1.jpg",0)
# ~ attractMask=img1*0
# ~ attractMask[690:810,100:200]=1
#img1=cv2.resize(img1,(0,0),fx=.4,fy=.4)
# ~ #img1=cv2.GaussianBlur(img1,(37,37),-1)
# ~ show(getEdgeImage(img1))
# ~ show(getEnergyMap(img1))

# ~ highlight= []
# ~ while 1:
    # ~ seam=getSeam(img1,attractMask=attractMask)
    # ~ img1=removeSeam(img1,seam)
    # ~ attractMask=removeSeam(attractMask,seam)
    # ~ show(img1,wait=0)

