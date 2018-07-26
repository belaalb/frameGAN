import cv2
import numpy as np
import math
import glob
import os
from tqdm import trange

def davis_gen(im_size=64,output_path="./results/"):
	
	if not os.path.exists(output_path):
	    os.makedirs(output_path)

	classesDir = os.listdir('./JPEGImages/480p')
	f = open(output_path+'log_file.txt', 'w')

	for v in trange(0,len(classesDir), desc='Progress in DAVIS Dataset'):

		emptyFlag = 0
		if not os.path.exists(output_path + classesDir[v]):
			os.makedirs(output_path + classesDir[v])
		#classesDir[v]
		filenames = glob.glob("./Annotations/480p/" + classesDir[v] + "/*.png")
		filenames.sort()
		images = [cv2.imread(file,0) for file in filenames]

		filenamesFull = glob.glob("./JPEGImages/480p/" + classesDir[v] + "/*.jpg")
		filenamesFull.sort()
		imagesFull = [cv2.imread(file,cv2.IMREAD_COLOR) for file in filenamesFull]

		images = np.asarray(images)
		imagesFull = np.asarray(imagesFull)

		imgSize = images[0].shape
		#print(imgSize)

		k = 0
		for p in trange(0,len(images),desc='Progress in ' + str(classesDir[v])):
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
					if(images[p,i,j] == 255):
						xLeft = min(i,xLeft)
						xRight = max(i,xRight)
						yUp = min(j,yUp)
						yDown = max(j,yDown)			

			x = int((xLeft+xRight)/2)
			y = int((yUp+yDown)/2)


			# get first masked value (foreground)
			imgFinal = cv2.bitwise_or(imagesFull[k], imagesFull[k], mask=images[p])

			crop_img = imgFinal[xLeft:xRight,yUp:yDown]

			outputName = str(filenames[k]).split('/')
			if(crop_img.shape[0] != 0):
				#print(crop_img.shape)
				crop_img = cv2.resize(crop_img,(im_size,im_size))
			else:
				crop_img = np.zeros((im_size,im_size,3))
				emptyFlag = 1
	    		
			cv2.imwrite(output_path + classesDir[v] + '/' + outputName[-1], crop_img)
			
			## DEBUG IMAGES GENERATED
			#print(output_path + classesDir[v] + '/'+ outputName[-1])
			#cv2.imshow("teste", crop_img)
			#cv2.waitKey(10)

			k = k+1
		if(emptyFlag == 1):
			f.write('Class ('+classesDir[v]+') There are empty frames\n')
	f.close()

def davis_count_frames_per_class(output_path="./results/"):
	
	if not os.path.exists(output_path):
	    os.makedirs(output_path)

	classesDir = os.listdir('./JPEGImages/480p')

	for v in range(0,len(classesDir)):

		emptyFlag = 0
		if not os.path.exists(output_path + classesDir[v]):
			os.makedirs(output_path + classesDir[v])
		#classesDir[v]
		filenames = glob.glob("./Annotations/480p/" + classesDir[v] + "/*.png")
		print(classesDir[v] + ': '+ str(len(filenames)))



if __name__ == '__main__':
	#davis_count_frames_per_class()
	
	import argparse
	
	# Data settings
	parser = argparse.ArgumentParser(description='Generate Flying Shapes Dataset')
	parser.add_argument('--im-size', type=int, default=64, metavar='N', help='H and W of frames (default: 64)')
	parser.add_argument('--output-path', type=str, default='./results/', metavar='Path', help='Path for output')
	args = parser.parse_args()

	dat = davis_gen(im_size=args.im_size, output_path=args.output_path)