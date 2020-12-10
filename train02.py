import cv2,os


data_path='D:/20201/AI/data/image_cut01'
categories=os.listdir(data_path)
labels=[i for i in range(len(categories))]
label_dict=dict(zip(categories,labels)) #empty dictionary
print(label_dict)
print(categories)
print(labels)

data=[]
target=[]
img_size = 100
for category in categories:
    folder_path=os.path.join(data_path,category)
    img_names=os.listdir(folder_path)
    for img_name in img_names:
        img_path=os.path.join(folder_path,img_name)
        img=cv2.imread(img_path)
        try:
            gray_1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            if(gray_1.shape[0] >= gray_1.shape[1]):
                gray = gray_1[0:int(gray_1.shape[0]/2),:]
            else:
                gray = gray_1[:,0:int(gray_1.shape[1]/2)]
            resized=cv2.resize(gray,(img_size,img_size))
            #resizing the gray scale into 100x100, since we need a fixed common size for all the images in the dataset
            data.append(resized)
            target.append(label_dict[category])
        except Exception as e:
            print('Exception:',e)
    cv2.imshow('LIVE',gray)
    cv2.waitKey()
import numpy as np
data=np.array(data)/255.0
data=np.reshape(data,(data.shape[0],img_size,img_size,1))
target=np.array(target)
from keras.utils import np_utils
target=np_utils.to_categorical(target)

from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
#Instantiate an empty model
model=Sequential()
model.add(Conv2D(200,(3,3),input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(100,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(50,activation='relu'))
model.add(Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

from sklearn.model_selection import train_test_split
train_data,test_data, train_target,test_target=train_test_split(data,target,test_size=0.1)

checkpoint = ModelCheckpoint('D:/20201/AI/data_test/hdf5/09121628_ep30_bs16_haft_butter_02.hdf5',monitor='val_loss',verbose=1,save_best_only=True,mode='min')
history=model.fit(train_data,train_target,epochs=30, batch_size= 16 ,callbacks=[checkpoint],validation_split=0.2)