import numpy as np
import cv2
import math

def show(img,wait=0,destroy=True):
    img=np.uint8(img)
    cv2.imshow("image",img)
    cv2.waitKey(wait)
    if destroy:
        cv2.destroyAllWindows()

pug=cv2.imread("input/image1.png")
nug=cv2.imread("input/image2.png")
cube=cv2.imread("input/image3.png")
puffs=cv2.imread("input/image4.png")

# ~ #reflected across the y axis
# ~ h,w=pug.shape[:2]
# ~ M= np.float64([[-1,0,w],[0,1,0],[0,0,1]])
# ~ out=cv2.warpPerspective(pug,M,(w,h))
# ~ cv2.imwrite("output\pic_1_1.png",np.uint8(out))
# ~ show(out)

# ~ #rotate 30 degrees from the bottom right corner
# ~ h,w=nug.shape[:2]
# ~ angle=math.pi*30/180
# ~ T1=np.float64([[1,0,-w],[0,1,-h],[0,0,1]])
# ~ T2=np.float64([[1,0,w],[0,1,h],[0,0,1]])
# ~ R=np.float64([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
# ~ M=T2@R@T1
# ~ nugR=cv2.warpPerspective(nug,M,(w,h))
# ~ show(nugR)
# ~ cv2.imwrite("output\pic_1_2.png", np.uint8(nugR))

#frames the image
# ~ h,w=pug.shape[:2]
# ~ angle=math.pi*30/180
# ~ T1=np.float64([[1,0,-w],[0,1,-h],[0,0,1]])
# ~ T2=np.float64([[1,0,w],[0,1,h],[0,0,1]])
# ~ R=np.float64([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])
# ~ M=T2@R@T1
# ~ pugR=cv2.warpPerspective(pug,M,(w,h))
# ~ show(pugR)

# ~ newX=(h*np.sin(angle))+(w*np.cos(angle))
# ~ newY=(h*np.cos(angle))+(w*np.sin(angle))
# ~ T3=np.float64([[1,0,-(newX-w)/(17/16)],[0,1,(newY-h)],[0,0,1]])
# ~ M2=T3@T2@R@T1
# ~ print(newX,newY)
# ~ pugR=cv2.warpPerspective(pug,M2,(int(newX),int(newY)))
# ~ show(pugR)
# ~ cv2.imwrite("output\pic_1_3.png", np.uint8(pugR))


# ~ #paste to side of cubes
hPug,wPug=pug.shape[:2]
hNug,wNug=nug.shape[:2]
cubeh,cubew=cube.shape[:2]
srcPointsPug=np.float32([[[0,0],[wPug,0],[wPug,hPug],[0,hPug]]])
srcPointsNug=np.float32([[[0,0],[wNug,0],[wNug,hNug],[0,hNug]]])
dstPointsPug=np.float32([[[625,335],[955,325],[925,640],[620,680]]])
dstPointsNug=np.float32([[[510,290],[370,300],[380,470],[515,440]]])
MPug=cv2.getPerspectiveTransform(srcPointsPug, dstPointsPug)
MNug=cv2.getPerspectiveTransform(srcPointsNug, dstPointsNug)


nugPerspective=cv2.warpPerspective(nug,MNug,(cubew,cubeh))
show(nugPerspective)
print(1)
blank=nug*0+255
blankPerspective=cv2.warpPerspective(blank,MNug,(cubew,cubeh))
p=blankPerspective/255.0
cube=np.uint8(nugPerspective*p+cube*(1-p))
show(cube)

pugPerspective=cv2.warpPerspective(pug,MPug,(cubew,cubeh))
show(pugPerspective)
blank=pug*0+255
blankPerspective=cv2.warpPerspective(blank,MPug,(cubew,cubeh))
p=blankPerspective/255.0
cube=np.uint8(pugPerspective*p+cube*(1-p))
show(cube)

cv2.imwrite("output\pic_1_4.png",np.uint8(cube))

#Flatten
# ~ blank=puffs*0
# ~ h,w=puffs.shape[:2]
# ~ srcPoints=np.float32([[[1030,65],[350,110],[355,1265],[1030,1190]]])
# ~ dstPoints=np.float32([[[0,0],[w,0],[w,h],[0,h]]])
# ~ M=cv2.getPerspectiveTransform(srcPoints, dstPoints)
# ~ flatPerspective=cv2.warpPerspective(puffs,M,(w,h))
# ~ blankPerspective=cv2.warpPerspective(puffs,M,(w,h))
# ~ p=blankPerspective/255.0
# ~ blank=np.uint8(flatPerspective*p+blank*(1-p))

# ~ h,w=blank.shape[:2]
# ~ M= np.float64([[-1,0,w],[0,1,0],[0,0,1]])
# ~ blank=cv2.warpPerspective(blank,M,(w,h))
# ~ show(blank)

# ~ cv2.imwrite("Output\pic_1_5_0.png",np.uint8(blank))

