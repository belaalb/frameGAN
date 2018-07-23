import cv2
import numpy as np
import math
import glob


filenames = glob.glob("*.png")
filenames.sort()
images = [cv2.imread(file,0) for file in filenames]

filenamesFull = glob.glob("/home/rapela/Downloads/davis/DAVIS/JPEGImages/480p/bear/*.jpg")
filenamesFull.sort()
imagesFull = [cv2.imread(file,cv2.IMREAD_COLOR) for file in filenamesFull]

#img = cv2.imread('00000.png',0)
images = np.asarray(images)
imagesFull = np.asarray(imagesFull)

imgSize = images[0].shape
print(imgSize)

k = 0
for img in images:
	#print(img.shape)
	x = 0.0
	y = 0.0
	imgFinal = np.zeros((imgSize[0],imgSize[1],3))

	xLeft = 999999
	xRight = -1

	yDown = -1
	yUp = 999999
	#print(imagesFull.shape)
	for i in range(0,imgSize[0]):
		for j in range(0,imgSize[1]):
			if(img[i,j] == 255):
				xLeft = min(i,xLeft)
				xRight = max(i,xRight)
				yUp = min(j,yUp)
				yDown = max(j,yDown)			

	x = int((xLeft+xRight)/2)
	y = int((yUp+yDown)/2)


	# get first masked value (foreground)
	imgFinal = cv2.bitwise_or(imagesFull[k], imagesFull[k], mask=img)

	#print(x,y)
	#print(xLeft,xRight,yUp,yDown)
	#cv2.rectangle(imgFinal, (yDown,xLeft), (yUp,xRight), (255,0,0), 2)


	#print(imgFinal.shape)
	#print(imgFinal.shape)

	#img = cv2.imread("lenna.png")
	crop_img = imgFinal[xLeft:xRight,yUp:yDown]
	#print(yUp,yDown, xLeft,xRight)
	#print(crop_img.shape)
	#cv2.imshow("cropped", crop_img)
	#cv2.waitKey(0)


	cv2.imwrite("./result/" + str(filenames[k]), crop_img)
	print("./result/" + str(filenames[k]))
	#cv2.imshow("teste", imgFinal)
	#cv2.imshow("teste2", imagesFull[k])
	#cv2.waitKey(10)

	#print(./result/" + str(filenames))

	k = k+1


#print(img)

#if()
#
#cv2.imshow("Teste", img)
#cv2.waitKey(0)