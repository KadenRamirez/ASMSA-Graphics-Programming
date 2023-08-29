import numpy as np
import cv2

image1=cv2.imread("input/image1.png")
image2=cv2.imread("input/image2.png")

#image1.png rg switch
# ~ red=image1[:,:,2]
# ~ green=image1[:,:,1]
# ~ image1_gr_switch=image1[:,:,:]
# ~ image1_gr_switch[:,:,1]=red
# ~ image1_gr_switch[:,:,2]=green

# ~ cv2.imwrite("output/pic_1_a.png",image1_gr_switch)

# image2[:,:,0]=0
cv2.imwrite("output/pic_1_b.png",image2)

cv2.imshow("Don't be Blue",image2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ~ greenR=image1[:,:,1]
# ~ image1[:,:,1]=255-greenR
# ~ cv2.imwrite("output/pic_1_c.png",image1)

# ~ image2+=100
# ~ image2[image2<100]=255
# ~ cv2.imwrite("output/pic_1_d.png",image2)

# ~ x,y,*w=image1.shape
# ~ print(image1.shape)
# ~ print(y//2)
# ~ image1[x//2-50:x//2+50,y//2-50:y//2+50,1]=255
# ~ cv2.imwrite("output/pic_2_a.png",image1)

# ~ x1,y1,*w=image1.shape
# ~ x2,y2,*w=image2.shape
# ~ cut=image1[x1//2-50:x1//2+50,y1//2-50:y1//2+50,:]
# ~ image2[x2//2-50:x2//2+50,y2//2-50:y2//2+50,:]=cut
# ~ cv2.imwrite("output/pic_2_b.png",image2)

x,y,*w=image1.shape
print("the number of pixels in image1 are " + str(x*y))
print(np.std(image1))
# ~ print(np.max(image1))

w = 1200
h = 720
triOffSet = 32.5
img=np.zeros((h, w, 3), dtype=np.uint8)

#colors
green=(78, 149, 9)
white=255
yellow=(22,209,252)
black=0
red=(38,17,206)

#green base
img[:,:]=green

#white triangle
Y,X=np.mgrid[:h,:w]
whiteTriangle=(Y>h/w*X/2+0)*(-h/w*X/2+h>Y)
img[whiteTriangle]= white

#yellow Triangle
yellowTriangle=(Y>h/w*X/2+triOffSet)*(-h/w*X/2+h-triOffSet>Y)
img[yellowTriangle]= yellow

#black Traingle
blackTriangle=(Y>h/w*X+0)*(-h/w*X+h>Y)
img[blackTriangle]= black

#red Traingle
redTriangle=(Y>h/w*X+triOffSet)*(-h/w*X+h-triOffSet>Y)
img[redTriangle]=red

#flag from scratch
cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#wiki page flag picture
flag=cv2.imread("input/GuyanaFlag.png")
cv2.imshow("Guyana",flag)
cv2.waitKey(0)
cv2.destroyAllWindows()

diff=flag*1.0-img
diff-=np.min(diff)
diff/=np.max(diff)
diff*=255
diff=np.uint8(diff)
# ~ print(np.max(diff),np.min(diff))

cv2.imwrite("output/pic_4_b.png",diff)

cv2.imshow("diff",diff)
cv2.waitKey(0)
cv2.destroyAllWindows()


