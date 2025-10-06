  #latest
##########################REQUIRED LIBRARIES#############################
# pip install numpy
# pip install opencv-python
# pip install matplotlib
# pip install scikit-learn
# pip install Pillow
#########################################################################
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pickle
import random
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

dir = "datasets_underwater_object\\train"
categories = ['fish', 'jellyfish', 'penguin','puffin', 'shark', 'starfish', 'stingray']
data = []

#########################################################################

for category in categories:
    path = os.path.join(dir, category)
    label = categories.index(category)


#preprocessing data#
    for img in os.listdir(path):
        imgpath = os.path.join(path,img)
        animals_img = cv2.imread(imgpath,0)
        try:
            animals_img = cv2.resize(animals_img,(50,50))
            image = np.array(animals_img).flatten()

            data.append([image, label])
        except Exception as e:
            pass
    

#pre-saved dataset, contains pairs of [image_vectors, label]
pick_in = open('data2.pickle', 'wb') 
data = pickle.load(pick_in)
pick_in.close()

random.shuffle(data)
features = []
labels = []

for feature, label in data:
    features.append(feature)
    labels.append(label)

xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.20)


model = SVC(C=1,kernel='poly',gamma='auto')
model.fit(xtrain, ytrain)

pick=open('MODEL.sav','wb')
#Once saved, it can be loaded without retraining
pickle.dump(model,pick)
pick.close()

