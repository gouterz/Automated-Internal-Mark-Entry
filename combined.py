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
from PIL import Image

def append_df_to_excel(df, excel_path):
    df_excel = pd.read_excel(excel_path)
    df_excel = df_excel[df_excel.filter(regex='^(?!Unnamed)').columns]
    result = pd.concat([df_excel, df], ignore_index=True,sort=False)
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
    mnist_model = load_model("/gdrive/My Drive/conclave gdrive final/mnist-model-combo-inv-more3.h5py")

    #check for any image
    img_classes = mnist_model.predict_classes(img)
    classname = img_classes[0]
    
    #print(classname,end='')
    regno += str(classname)
    return regno

def find_mark(img):
    img = cv2.resize(img ,(28 ,28) ,interpolation = cv2.INTER_AREA)
    img = img.reshape(-1 ,img.shape[0] ,img.shape[1] ,1)
    img = img.astype('float64')
    img /=255

    # load from saved training from the next run
    mnist_model = load_model("/gdrive/My Drive/conclave gdrive final/mnist-model-bn-combo-inv.h5py")

    # check for any image
    img_classes = mnist_model.predict_classes(img)
    classname = img_classes[0]
    return classname

def prediction(i): 
    im = cv2.imread('/gdrive/My Drive/conclave gdrive final/Cropped Image Dataset/Cropped paper1/Register box '+str(i+1)+'.jpg',0)

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
        (x,y,w,h) = cv2.boundingRect(c)
        if (h>=45 and h<=80) and (w>=45 and w<=80):
            cv2.rectangle(thresh,(x,y),(x+w,y+h),(0,255,0),2)
            roi = thresh[y+2:y+h-2, x+2:x+w-2]
        #roismall = image_resize(roi,28,28)
            roismall = cv2.resize(roi,(28,28),fx=0.5,fy=0.5)
            regno = find(regno,roismall)
            #print(roismall.shape)
            #cv2.imshow("I",image)
        #print(regno)
        else:
            j=1

    print("Register Number: ",regno)
    im = cv2.imread('/gdrive/My Drive/conclave gdrive final/Cropped Image Dataset/Cropped paper1/Mark box '+str(i+1)+'.jpg',0)

    retr ,thresh = cv2.threshold(im ,120,255 ,cv2.THRESH_BINARY_INV)
    # thresh = cv2.equalizeHist(thresh)

    # erode dilate an morphology to remove noise
    kernel = np.ones((1 ,1) ,'int32')
    thresh = cv2.dilate(thresh ,kernel ,iterations = 1)
    thresh = cv2.erode(thresh ,kernel ,iterations = 1)
    # thresh = cv2.morphologyEx(thresh,cv2.MORPH_OPEN, kernel)

    image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    (contours ,bounding_box) = sort_contours(contours ,method="top-to-bottom")

    # draw contours
    image = cv2.cvtColor(image ,cv2.COLOR_GRAY2RGB)

    areas = [cv2.contourArea(c) for c in contours]
    max_index = np.argmax(areas)
    cnt = contours[max_index]

    (x ,y ,w ,h) = cv2.boundingRect(cnt)

    parta =['0' for i in range(0,10) ]
    partb=['0' for i in range(0,10)]
    aindex=0
    bindex=0
    print('Processing and identifying marks')
    for c in contours:
        flag=0
        (x ,y ,w ,h) = cv2.boundingRect(c)
        if x>=200 and y>=60 and ( w>=160 and w<= 260) and ((h >= 40 and h <= 70) or (h >= 100 and h <= 120)):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            roi = thresh[y + 2:y + h - 2, x + 2:x + w - 2]
            roismall = cv2.resize(roi, (0, 0), fx=0.5, fy=0.5)
            #find(roismall)
            roicrop, cnt, hier = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(cnt)>0:
                (cnt ,bounding_box) = sort_contours(cnt ,method="left-to-right")
            if x>=450:
                #print('\t',end='')
                flag = 1
            if len(cnt) > 0:
                string=""
                for cn in cnt:
                # maxCnt = max(cnt,key=cv2.contourArea)
                #print(cv2.contourArea(cn))
                #if (cv2.contourArea(cn) >= 180 and cv2.contourArea(cn) <= 900)or(w>=15 and h>=30 and h<=40):
                    (x, y, w, h) = cv2.boundingRect(cn)
                    if w>=8 and h>=20 :
                    #print ('w',w,'h',h)
                        roismall = roicrop[y:y + h, x :x + w]
                        roismall = cv2.copyMakeBorder(roismall,2,2,2,2,cv2.BORDER_REPLICATE)
                        num = find_mark(roismall)
                        string += str(num)
                if(flag):
                    partb[bindex]=(string)
                    bindex+=1
                    flag =0
                else:
                    parta[aindex]=(string)
                    aindex+=1
    print(parta)
    print(partb)


    column = ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8','Q9','Q10','Q11a','Q11b','Q12a','Q12b','Q13a','Q13b','Q14a','Q14b','Q15a','Q15b']


    df = pd.DataFrame({"Register Number":[regno],column[0]:parta[0],column[1]:parta[1],column[2]:parta[2],column[3]:parta[3],column[4]:parta[4],column[5]:parta[5],column[6]:parta[6],column[7]:parta[7],column[8]:parta[8],column[9]:parta[9],column[10]:partb[0],column[11]:partb[1],column[12]:partb[2],column[13]:partb[3],column[14]:partb[4],column[15]:partb[5],column[16]:partb[6],column[17]:partb[7],column[18]:partb[8],column[19]:partb[9]})
    df = df[df.filter(regex='^(?!Unnamed)').columns]
    append_df_to_excel(df, r"sample.xls")


        
