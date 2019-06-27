import numpy as np
import cv2
from PIL import Image
from keras.models import load_model

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
                                        key=lambda b :b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def find_mark(img):
    img = cv2.resize(img ,(28 ,28) ,interpolation = cv2.INTER_AREA)
    img = img.reshape(-1 ,img.shape[0] ,img.shape[1] ,1)
    img = img.astype('float64')
    img /=255

    # load from saved training from the next run
    mnist_model = load_model("mnist-model-bn-combo-inv.h5py")

    # check for any image
    img_classes = mnist_model.predict_classes(img)
    classname = img_classes[0]
    return classname

im = cv2.imread('C:/Users/sakth/Documents/InternalPaper/Cropped Internal Paper/Mark box 4.png',0)
'''
im = Image.open('C:/Users/sakth/Documents/InternalPaper/Cropped Internal Paper/Mark box 5.png')
thresh = im.convert('1')
'''
'''
#blur to remove noise
im = cv2.GaussianBlur(im,(5,5),0)
retr, thresh = cv2.threshold(im,200,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
'''

# set threshold value
# thresh = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,2)
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

x ,y ,w ,h = cv2.boundingRect(cnt)

parta =[]
partb=[]
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
                x, y, w, h = cv2.boundingRect(cn)
                if w>=8 and h>=20 :
                    #print ('w',w,'h',h)
                    roismall = roicrop[y:y + h, x :x + w]
                    roismall = cv2.copyMakeBorder(roismall,2,2,2,2,cv2.BORDER_REPLICATE)
                    num = find_mark(roismall)
                    string += str(num)
            if(flag):
                partb.append(string)
                flag =0
            else:
                parta.append(string)
print(parta)
print(partb)
cv2.imshow('image',image)
cv2.waitKey(0)
