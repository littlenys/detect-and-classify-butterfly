import cv2
import sys 
from keras.models import load_model
import cv2
import numpy as np
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--hdf5", required=True,
	help="path to .hdf5 file")
ap.add_argument("-x", "--xml", required=True,
	help="path to .xml file")
ap.add_argument("-v", "--image", required=True,
	help="path to image")
args = vars(ap.parse_args())
#python d:/20201/AI/detect_02.py --hdf5 D:/20201/AI/result/weights01.hdf5 --xml D:/20201/AI/butterfly_cascade.xml --image D:/20201/AI/leedsbutterfly/images/0070112.PNG

model = load_model(args["hdf5"])
cascPath = args["xml"]
source = cv2.imread(args["image"])
labels_dict={0:'Danaus_plexippus', 1:'Heliconius_charitous', 2:'Heliconius_erato', 3:'Junonia_coenia',4: 'Lycaena_phlaeas', 5:'Nymphalis_antiopa', 6:'Papilio_cresphontes', 7:'pieris_rapae',8: 'Vanessa_atalanta',9: 'Vanessa_cardui'}
color_dict={0:(0,255,0),1:(0,0,255),2:(225,0,0),3:(255,255,0),4:(0,255,255),5:(255,0,255),6:(0,150,150),7:(150,150,150),8:(150,150,0),9:(150,0,150)}


 
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)


img = source
gray = img
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=15,
    minSize=(30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)
i = 0
for (x,y,w,h) in faces :
    i = 1
    '''a = int(w/10)
    x = x- a
    y = y- a
    w= w+2*a
    h = h + 2*a'''
    face_img = gray[y:y+w, x:x+w]
    resized = cv2.resize(face_img,(100,100))
    normalized = resized/255.0
    reshaped = np.reshape(normalized,(1,100,100,3))
    result = model.predict(reshaped)

    label = np.argmax(result,axis = 1)[0]

    cv2.rectangle(img , (x,y) , (x+w , y+h), color_dict[label], 2)
    cv2.rectangle(img , (x, y-40) , (x+w,y) , color_dict[label],-1)
    cv2.putText(img, str(label)+" : "+labels_dict[label],(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8 , (255,255,255),2)
    print(labels_dict[label])

cv2.namedWindow("LIVE",cv2.WINDOW_GUI_NORMAL)

if (i == 0):
    x = 50
    y = 50
    w = 200
    h = 200
    cv2.rectangle(img , (x, y-40) , (x+w,y) , (0,0,255),-1)
    cv2.putText(img, "Not found !!!",(x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.8 , (255,255,255),2)
    cv2.imshow('LIVE',img)
    cv2.waitKey()
else:
    cv2.imshow('LIVE',img)
    cv2.waitKey()

cv2.destroyAllWindows()










'''
import glob
import cv2
from PIL import Image
import numpy as np
from keras.models import load_model
from keras import Sequential
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt

# Get user supplied values
imagePath = "D:/aaa1.jpg"
cascPath = "D:/MITECH/task5/02_resources/022_model/haarcascade_frontalface_default.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(50, 50),
    flags = cv2.CASCADE_SCALE_IMAGE
)

print( "Found {0} faces!".format(len(faces)))
t_image_array=[]
t = 0
bounding_box_x = []
bounding_box_y = []
# Draw a rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x-10, y-10), (x+w+10, y+h+10), (0, 255, 0), 2)
    crop_image= image[y-10:y+h+10, x-10:x+w+10]
    image_from_array = Image.fromarray(image, 'RGB')
    size_image = image_from_array.resize((50,50))
    t_image_array.append(np.array(size_image))
    t = t+1
    bounding_box_x.append(x)
    bounding_box_y.append(y)



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

for img in glob.glob("D:/MITECH/task5/02_resources/021_data/dataset/woman/*.JPG"):
    image= cv2.imread(img)
    image_from_array = Image.fromarray(image, 'RGB')
    size_image = image_from_array.resize((50,50))
    t_image_array.append(np.array(size_image))
    t = t+1

model = Sequential()
model = load_model("D:/MITECH/task5/02_resources/022_model/weights04.hdf5")
data1=np.array(t_image_array)
np.save("image1.npy",data1)
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
        cv2.putText(image,'Nobeard',(bounding_box_x[i],bounding_box_y[i]-10), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)

    elif (prediction[i] == 'barbate'):
        barbate = barbate +1
        cv2.putText(image,'beared',(bounding_box_x[i],bounding_box_y[i]-10), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)

    elif (prediction[i] == 'woman'):
        woman = woman +1
        cv2.putText(image,'woman',(bounding_box_x[i],bounding_box_y[i]-10), cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)


cv2.namedWindow("Faces found",cv2.WINDOW_GUI_NORMAL)
cv2.imwrite("D:/DetectAndClassification.jpg",image)
cv2.imshow("Faces found", image)
cv2.waitKey(0)


print(t," face")
print(nobeard," nobeard ")
print(barbate," barbate")
print(woman," woman")'''