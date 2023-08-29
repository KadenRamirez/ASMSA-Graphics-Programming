import numpy as np
import cv2

#normalize
def normalize(img):
	img=img-np.min(img)
	img=img/np.max(img)
	img*=255.99
	return np.uint8(img)

#shows the image
def show(img,wait=0,destroy=True):
	if np.min(img)<0 or np.max(img)>255:
		img=normalize(img)
		print("had to normalize for viewing")
	cv2.imshow("image", np.uint8(img))
	cv2.waitKey(wait)
	if destroy:
		cv2.destroyAllWindows()

def greyscale(img,s=False):
	v=img*1.0
	grey=img*1.0
	b,g,r= v[:,:,0], v[:,:,1], v[:,:,2]
	grey=b*.1 + g*.7 + r*.2
	if s==True:
		show(grey)
		return grey
	return grey

#changes the image to black and white
# depeding on a threshold
def blackWhite(img,threshold=128,s=False):
	bl=img[:,:,1]*1.0
	bl[bl<=threshold]=0
	bl[bl>threshold]=255
	if s==True:
		show(bl)
		return bl
	return bl

#desaturates the image by half unless stated otherwise
def desaturate(img, percent=1,s=False):
	img2=img*1.0
	if percent==0:
		if s==True:
			show(img2)
			return img2
		return img2
	if percent==1:
		if s==True:
			show(greyscale(img2))
			return greyscale(img2)
		return img2
	sat=img[:,:,1]
	grey=img*1.0
	grey[:,:,0]=sat
	grey[:,:,1]=sat
	grey[:,:,2]=sat
	img2=(img*(1-percent) + grey*percent)
	if s==True:
		show(img2)
		return img2
	return img

def contrast(img,factor=1,s=False):
	img2=img*1.0
	np.uint8(img2)
	img2=(img2-128)*factor+128
	img2[img2<0]=0
	if s==True:
		show(img2)
		return img2
	return img2

def tint(img,color,percent=0.5,s=False):
	tinter=int(255*percent)
	imgT=img*1.0
	r=imgT[:,:,color]
	r=(1-percent)*r+percent*tinter
	imgT[:,:,color] = r
	if s==True:
		show(imgT)
		return imgT
	return imgT

img=cv2.imread("Input/image1.jpg")
# ~ cv2.imwrite("Output/pic_2_1.png",greyscale(img,s=True))

# ~ #black and white image writing
# ~ cv2.imwrite("Output/pic_2_2.png",blackWhite(img,128,s=True))
# ~ for i in range(8):
	# ~ thresh=(i+1)*32
	# ~ np.uint8(thresh)
	# ~ blackWhite(img,thresh)
	# ~ cv2.imwrite("output/pic_2_2_"+str(i)+".png",blackWhite(img,(i+1)*32,s=True))

#desaturation image writing
for i in range(11):
	print(i)
	img=desaturate(img,(i))
	show(img)
	# ~ cv2.imwrite("Output/pic_2_3_"+str(i)+".png",desaturate(img,(i)*.1,s=True))
	
# ~ for i in range(11):
	# ~ print(i)
	# ~ cv2.imwrite("Output/pic_2_4_"+str(i)+".png",contrast(img,(i+5)*.1,s=True))

# ~ for i in range(3):
	# ~ cv2.imwrite("Output/pic_2_5_"+str(i)+".png",tint(img,color=i,percent=.7,s=True))

