from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
import keras
from matplotlib import pyplot as plt
import cv2
from keras.models import load_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from keras.utils import np_utils
import os
from PIL import Image
import numpy as np
from array import array
from sklearn.utils import shuffle


batch_size = 32
num_classes = 10
epochs = 1
img_h, img_w = 28,28

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], img_h, img_w, 1)
X_test = X_test.reshape(X_test.shape[0],img_h, img_w,1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape)
print(Y_train.shape)

Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)


#new code
'''
path1 = 'C:/Users/Sakth/Documents/InternalPaper/Training set'


listing = os.listdir(path1)
num_samples = len(listing)
print(num_samples)


for file in listing:
    im = Image.open(path1 + '//'+file)
    img = im.resize((28,28))
    gray = img.convert('L')
    gray.save(path2+'/'+file,"jpeg")
'''
path2 = 'C:/Users/sakth/Documents/InternalPaper/temp training/data'

imlist = os.listdir(path2)

#im1 = np.array(Image.open('C:/Users/sakth/Documents/InternalPaper/new training/training set resized'+'//'+imlist[0]))
#m,n = im1.shape[0:2]
#imn = len(imlist)

immatrix = np.array([np.array(Image.open(path2+'/'+im2)) for im2 in imlist],'f')
'''
label = np.ones((imn,),dtype=int)
label[0:271] = 0
label[272:665] = 1
label[666:696] = 2
label[697:745] = 3
label[746:880] = 4
label[881:1140] = 5
label[1141:1290] = 6
label[1291:1416] = 7
label[1417:1433] = 8
label[1433:1446]=9
'''

label1 = np.ones((len(imlist),),dtype = int)
label1[0:258] = 0
label1[258:664] = 1
label1[664:690] = 2
label1[690:749] = 3
label1[749:895] = 4
label1[895:1176] = 5
label1[1176:1315] = 6
label1[1315:1438] = 7
label1[1438:1448] = 8
label1[1448:1463]=9

'''
print('Here')
j=0
imlist=os.listdir(path2)
for i in imlist:
    img=cv2.imread(path2+'/'+i)
    print(label1[j])
    cv2.imshow('',img)
    cv2.waitKey(0)
    if(input()=='Y'):
        label1[j]=int(input())
    j+=1
    
print(np.shape(label),np.shape(label1))

immatrix = np.concatenate((immatrix,immatrix1),axis=0)
label = np.concatenate((label,label1),axis=0)
'''
print(np.shape(immatrix),np.shape(label1))

data,label = shuffle(immatrix,label1,random_state = 2)
train_data = [data,label1]
(X,y) = (train_data[0],train_data[1])

X_train2,X_test2,y_train2,y_test2 = train_test_split(X,y, test_size=0.2, random_state = 4)

X_train2 = X_train2.reshape(X_train2.shape[0],28,28,1)
X_test2 = X_test2.reshape(X_test2.shape[0],28,28,1)

X_train2 = X_train2.astype('float32')
X_test2 = X_test2.astype('float32')

X_train2 /=255
X_test2 /= 255
Y_train2 = np_utils.to_categorical(y_train2, 10)
Y_test2 = np_utils.to_categorical(y_test2,10)

X_train = np.concatenate((X_train, X_train2), axis=0)
Y_train = np.concatenate((Y_train, Y_train2), axis=0)

X_test=np.concatenate((X_test,X_test2),axis=0)
Y_test=np.concatenate((Y_test,Y_test2),axis=0)


model = Sequential()
model.add(Conv2D(32,(3,3),activation = 'tanh', input_shape = X_train.shape[1:]))
model.add(Conv2D(64,(3,3),activation = 'tanh'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Conv2D(128,(3,3),activation='tanh'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation = 'softmax'))

model.compile(loss = keras.losses.categorical_crossentropy,
              optimizer = keras.optimizers.Adagrad(),
              metrics = ['accuracy'])

model.fit(X_train, Y_train,validation_data=(X_test,Y_test),
          batch_size=batch_size,
          epochs=epochs,
          verbose = 1)
'''
model.save("model-combo-inv-more3-adagradOpti.h5py")
X_test = np.concatenate((X_test,X_test2),axis=0)
Y_test = np.concatenate((Y_test,Y_test2),axis=0)
'''

mnist_model = load_model("model-combo-inv-more3-adagradOpti.h5py")

'''
score = mnist_model.evaluate(X_test, Y_test, batch_size=batch_size)
print(score[1])

'''
