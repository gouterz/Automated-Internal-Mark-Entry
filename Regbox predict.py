import cv2
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
#from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import load_model
from keras.preprocessing import image
from skimage.transform import resize
from sklearn import datasets
from sklearn.svm import SVC
from scipy import misc
import xlwt

def append_df_to_excel(df, excel_path):
    df_excel = pd.read_excel(excel_path)
    df_excel = df_excel[df_excel.filter(regex='^(?!Unnamed)').columns]
    result = pd.concat([df_excel, df], ignore_index=True)
    result.to_excel(excel_path, index=False)
def sort_contours(cnts, method="left-to-right"):
	# initialize the reverse flag and sort index
	reverse = False
	i = 0
 
	# handle if we need to sort in reverse
	if method == "right-to-left" or method == "bottom-to-top":
		reverse = True
 
	# handle if we are sorting against the y-coordinate rather than
	# the x-coordinate of the bounding box
	if method == "top-to-bottom" or method == "bottom-to-top":
		i = 1
 
	# construct the list of bounding boxes and sort them from top to
	# bottom
	boundingBoxes = [cv2.boundingRect(c) for c in cnts]
	(cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
		key=lambda b:b[1][i], reverse=reverse))
 
	# return the list of sorted contours and bounding boxes
	return (cnts, boundingBoxes)

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def find(regno,img):
    img = image_resize(img,28,28)
    img = img.reshape(1,img.shape[0],img.shape[1],1)
    img = img.astype('float64')
    img /=255
    
    #load from saved training from the next run
    mnist_model = load_model("mnist-model-combo-inv-more3.h5py")

    #check for any image
    img_classes = mnist_model.predict_classes(img)
    classname = img_classes[0]
    
    #print(classname,end='')
    regno += str(classname)
    return regno

        
im = cv2.imread('C:/Users/sakth/Documents/InternalPaper/temp training/Cropped paper/Register box 150.jpg',0)

#set threshold value
#thresh = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,115,1)
retr,thresh = cv2.threshold(im,127,255,cv2.THRESH_BINARY_INV)
#thresh = cv2.equalizeHist(thresh)

image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

(contours,bounding_box) = sort_contours(contours,method="left-to-right")

#draw contours
image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
regno = ""
print('Processing Register Number')
for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    if (h>=45 and h<=80) and (w>=45 and w<=80):
        cv2.rectangle(thresh,(x,y),(x+w,y+h),(0,255,0),2)
        roi = thresh[y+2:y+h-2, x+2:x+w-2]
        #roismall = image_resize(roi,28,28)
        roismall = cv2.resize(roi,(28,28),fx=0.5,fy=0.5)
        regno = find(regno,roismall)
        #print(roismall.shape)
        #cv2.imshow("I",image)
cv2.imshow('',thresh)
#print(regno)


df = pd.DataFrame({"Register":[regno]})
df = df[df.filter(regex='^(?!Unnamed)').columns]
append_df_to_excel(df, r"sample.xls")

