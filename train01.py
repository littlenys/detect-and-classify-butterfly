# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

categories = []
filenames = os.listdir("D:/MITECH/task5/02_resources/021_data/03/img_align_celeba/bearded")
for filename in filenames:
        #category = filename.split(".")[0]
        categories.append('001')

filenames = os.listdir("D:/MITECH/task5/02_resources/021_data/03/img_align_celeba/nobearded")
for filename in filenames:
        #category = filename.split(".")[0]
        categories.append('002')

filenames = os.listdir("D:/MITECH/task5/02_resources/021_data/03/img_align_celeba/woman")
for filename in filenames:
        #category = filename.split(".")[0]
        categories.append('003')

cat=pd.DataFrame(categories)
cat[0]=cat[0].replace({'001': 'beared', '002': 'nobeard', '003': 'woman'})
cat.head()

from sklearn.preprocessing import LabelEncoder
la=LabelEncoder()
labels=la.fit_transform(cat[0])
types=np.unique(labels)
types

import glob
import cv2
from PIL import Image
import numpy as np
image_array=[]
for img in glob.glob("D:/MITECH/task5/02_resources/021_data/03/img_align_celeba/bearded/*.JPG"):
    image= cv2.imread(img)
    image_from_array = Image.fromarray(image, 'RGB')
    size_image = image_from_array.resize((112,112))
    image_array.append(np.array(size_image))

for img in glob.glob("D:/MITECH/task5/02_resources/021_data/03/img_align_celeba/nobearded/*.JPG"):
    image= cv2.imread(img)
    image_from_array = Image.fromarray(image, 'RGB')
    size_image = image_from_array.resize((112,112))
    image_array.append(np.array(size_image))
for img in glob.glob("D:/MITECH/task5/02_resources/021_data/03/img_align_celeba/woman*.JPG"):
    image= cv2.imread(img)
    image_from_array = Image.fromarray(image, 'RGB')
    size_image = image_from_array.resize((112,112))
    image_array.append(np.array(size_image))


images=np.array(image_array)
np.save("image",images)
np.save("labels",labels)

image=np.load("image.npy",allow_pickle=True)
labels=np.load("labels.npy",allow_pickle=True)

s=np.arange(image.shape[0])
np.random.shuffle(s)
image=image[s]
labels=labels[s]

num_classes=len(np.unique(labels))
len_data=len(image)

x_train,x_test=image[(int)(0.1*len_data):],image[:(int)(0.1*len_data)]
y_train,y_test=labels[(int)(0.1*len_data):],labels[:(int)(0.1*len_data)]

import keras
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categorical(y_test,num_classes)

from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.layers import Activation, Convolution2D, Dropout
from keras.layers import AveragePooling2D, BatchNormalization
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential,Model
from keras.layers import Flatten , Dropout, Dense, Input, MaxPooling2D, Conv2D
from keras.callbacks import ModelCheckpoint
from keras import layers
from keras.regularizers import l2
import pandas as pd
import cv2

model = Sequential()
#
# 1st Convolutional Layer
model.add(Conv2D(filters=48, input_shape=(112,112,3), kernel_size=(7,7),strides=(2,2), padding='valid'))
model.add(Activation('relu'))
# Pooling 
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation before passing it to the next layer
model.add(BatchNormalization())
#
# 2nd Convolutional Layer
model.add(Conv2D(filters=128, kernel_size=(7,7), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())
#
# 3rd Convolutional Layer
model.add(Conv2D(filters=192, kernel_size=(5,5), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())
#
# 4th Convolutional Layer
model.add(Conv2D(filters=192, kernel_size=(3,3), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Batch Normalisation
model.add(BatchNormalization())
#
# 5th Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), padding='valid'))
model.add(Activation('relu'))
# Pooling
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), padding='valid'))
# Batch Normalisation
model.add(BatchNormalization())
#
# Passing it to a dense layer
model.add(Flatten())
# 1st Dense Layer
model.add(Dense(2048, input_shape=(112*112*3,)))
model.add(Activation('relu'))
# Add Dropout to prevent overfitting
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())
#
# 2nd Dense Layer
model.add(Dense(2048))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())
#
# 3rd Dense Layer
model.add(Dense(500))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# Batch Normalisation
model.add(BatchNormalization())
#
# Output Layer
model.add(Dense(2))
model.add(Activation('softmax'))
#
model.summary()
#
# (4) Compile 
model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])
#
# (5) Train

filepath="D:/MITECH/task5/02_resources/022_model/weights07.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
history=model.fit(x_train,y_train,batch_size=2,epochs=20,verbose=1,validation_split=0.33,callbacks=[checkpoint])
