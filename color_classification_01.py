

import glob
import cv2
from PIL import Image
import numpy as np
from keras.models import load_model
from keras import Sequential
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt

labels_dict={0:'Danaus_plexippus', 1:'Heliconius_charitous', 2:'Heliconius_erato', 3:'Junonia_coenia',4: 'Lycaena_phlaeas', 5:'Nymphalis_antiopa', 6:'Papilio_cresphontes', 7:'pieris_rapae',8: 'Vanessa_atalanta',9: 'Vanessa_cardui'}

i = 0
Danaus_plexippus = 0
Heliconius_charitous = 0
Heliconius_erato = 0
Junonia_coenia = 0
Lycaena_phlaeas = 0
Nymphalis_antiopa = 0
Papilio_cresphontes = 0
pieris_rapae = 0
Vanessa_atalanta = 0
Vanessa_cardui = 0
model = load_model("D:/20201/AI/data_test/hdf5/weights01.hdf5")
for img in glob.glob("D:/20201/AI/data_test/image/Danaus_plexippus/*.JPG"):
    image = cv2.imread(img)
    gray = image
    face_img = gray[:,:]
    resized = cv2.resize(face_img,(100,100))
    normalized = resized/255.0
    reshaped = np.reshape(normalized,(1,100,100,3))
    result = model.predict(reshaped)
    label = np.argmax(result,axis = 1)[0]
    i = i + 1
    if (label == 0 ):
        Danaus_plexippus = Danaus_plexippus +1
    elif (label == 1 ):
        Heliconius_charitous =  Heliconius_charitous +1
    elif (label == 2 ):
        Heliconius_erato  = Heliconius_erato+1
    elif (label == 3 ):
        Junonia_coenia = Junonia_coenia+1
    elif (label == 4 ):
        Lycaena_phlaeas = Lycaena_phlaeas+1
    elif (label == 5 ):
        Nymphalis_antiopa = Nymphalis_antiopa+1
    elif (label == 6 ):
        Papilio_cresphontes = Papilio_cresphontes +1
    elif (label == 7 ):
        pieris_rapae = pieris_rapae +1
    elif (label == 8 ):
        Vanessa_atalanta = Vanessa_atalanta +1 
    else:
        Vanessa_cardui = Vanessa_cardui +1

print(i," image")
print(Danaus_plexippus, "Danaus_plexippus")
print(Heliconius_charitous , "Heliconius_charitous")
print(Heliconius_erato , "Heliconius_erato")
print(Junonia_coenia , "Junonia_coenia")
print(Lycaena_phlaeas ,"Lycaena_phlaeas")
print(Nymphalis_antiopa ,"Nymphalis_antiopa")
print(Papilio_cresphontes ,"Papilio_cresphontes")
print(pieris_rapae ,"pieris_rapae")
print(Vanessa_atalanta ,"Vanessa_atalanta")
print(Vanessa_cardui , "Vanessa_cardui")