import cv2
import numpy as np
import math
import glob
import os
from tqdm import trange


def gif_gen(im_size=64,input_path="./results/", output_path='./davis_gif/'):

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    classesDir = os.listdir(input_path)
    vcat = []
    
    total = []
    aux_index = 0
    INPUT_SIZE = 20
    FRAMES = 40
    for v in trange(0,INPUT_SIZE, desc='Progress in DAVIS Dataset'):

        filenames = glob.glob(input_path + classesDir[v] + "/*.png")
        filenames.sort()
        #print(filenames)
        images = [cv2.imread(filenames[i],cv2.IMREAD_COLOR) for i in range(0,FRAMES)]
        
        images = np.asarray(images)
        
        imgSize = images[0].shape

        total.append(images)
    total = np.asarray(total)
    
    for j in range(0,FRAMES):
        hcat = []
        for inputNumber in range(0,INPUT_SIZE):
            black = [255,255,255]     #---Color of the border---       
            #print(images.shape)
            
            constant=cv2.copyMakeBorder(total[inputNumber,j,:,:,:],2,1,1,2,cv2.BORDER_CONSTANT,value=black )
            
            cv2.imshow('constant',constant)
            #cv2.waitKey(0)
            if(hcat == []):
                hcat = constant
            else:
                hcat = cv2.hconcat((hcat, constant))
        cv2.imwrite(output_path+str(j)+".png",hcat)
        #cv2.imshow('Final', hcat)
        #cv2.waitKey(0)
    
    cv2.destroyAllWindows()

if __name__ == '__main__':
    gif_gen()

