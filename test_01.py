

import glob
import cv2
from PIL import Image
import numpy as np
from keras.models import load_model
from keras import Sequential
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt

categories = []
categories.append('001')
categories.append('002')
categories.append('003')
cat=pd.DataFrame(categories)
cat[0]=cat[0].replace({'001': 'barbate', '002': 'nobeard', '003': 'woman'}) 
cat.head()

from sklearn.preprocessing import LabelEncoder
la=LabelEncoder()
labels=la.fit_transform(cat[0])
types=np.unique(labels)
types
t=0
t_image_array=[]
for img in glob.glob("D:/MITECH/task5/02_resources/021_data/03/img_align_celeba/bearded/*.JPG"):
    image= cv2.imread(img)
    image_from_array = Image.fromarray(image,'RGB')
    size_image = image_from_array.resize((100,100))
    t_image_array.append(np.array(size_image))
    t = t+1
model = Sequential()
model = load_model("D:/MITECH/task5/02_resources/022_model/weights22.hdf5")
data1=np.array(t_image_array)
np.save("image1",data1)
image1=np.load("image1.npy")
pred=np.argmax(model.predict(image1),axis=1)
prediction = la.inverse_transform(pred)

t_image=np.expand_dims(image1[t-1],axis=0)
pred_t=np.argmax(model.predict(t_image),axis=1)
prediction_t = la.inverse_transform(pred_t)
nobeard = 0
barbate = 0
woman = 0
for i in range (t):
    if (prediction[i] == 'nobeard'):
        nobeard = nobeard +1
    if (prediction[i] == 'barbate'):
        barbate = barbate +1
    if (prediction[i] == 'woman'):
        woman = woman +1

print(t," face")
print(nobeard," nobeard ")
print(barbate," barbate")
print(woman," woman")