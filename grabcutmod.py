import numpy as np
import cv2
import os, os.path
import imutils

imageDir = "/gdrive/My Drive/conclave gdrive final/Image Datasets/"
imageDir2 = "/gdrive/My Drive/conclave gdrive final/Cropped Image Dataset/"

destfolder = "Cropped paper1"
if not os.path.exists(imageDir2+destfolder):
        os.makedirs(imageDir2+destfolder)

i=1
#Creating output folder
image_path_list=[]
for file in os.listdir(imageDir):
    image_path_list.append(os.path.join(imageDir, file))

    
for imagePath in image_path_list:
    imgo = cv2.imread(imagePath)
    if imgo is None:
        continue

    imgo = cv2.resize(imgo, (1660, 2340))     
    if not os.path.exists(imageDir2+destfolder):
        os.makedirs(destfolder)

    crop_img = imgo[600:800, 10:1000]
    #crop_img = cv2.resize(crop_img, (900, 200))
    #cv2.imshow('',crop_img)
    cv2.imwrite(os.path.join(imageDir2+destfolder,"Register box "+str(i)+".jpg"),crop_img)
    cv2.waitKey(0)
    crop_img = imgo[1100:9000, 100:1800]
    cv2.imwrite(os.path.join(imageDir2+destfolder,"Mark box "+str(i)+".jpg"),crop_img)
    cv2.waitKey(0)
    i = i+1
    key = cv2.waitKey(0)
    if key==27:
        break
print("Cropping Complete")
cv2.destroyAllWindows()
