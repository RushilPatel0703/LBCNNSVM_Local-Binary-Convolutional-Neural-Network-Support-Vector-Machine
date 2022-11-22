import numpy as np
from skimage.feature import local_binary_pattern as lbp
from glob import glob
import cv2
import os

# This script loads images from a directory into seperate arrays
# then iterates over every image for every class to produce Local Binary Patterns dataset
angry = []
disgust = []
fear = []
happy = []
neutral = []
sad = []
surprise = []

path1 = 'C:/Users/Rushil/Desktop/BSC Computer Science/Honours/Research Project/tensorflow/images/emotions/angry/*.jpg'
path2 = 'C:/Users/Rushil/Desktop/BSC Computer Science/Honours/Research Project/tensorflow/images/emotions/disgust/*.jpg'
path3 = 'C:/Users/Rushil/Desktop/BSC Computer Science/Honours/Research Project/tensorflow/images/emotions/fear/*.jpg'
path4 = 'C:/Users/Rushil/Desktop/BSC Computer Science/Honours/Research Project/tensorflow/images/emotions/happy/*.jpg'
path5 = 'C:/Users/Rushil/Desktop/BSC Computer Science/Honours/Research Project/tensorflow/images/emotions/neutral/*.jpg'
path6 = 'C:/Users/Rushil/Desktop/BSC Computer Science/Honours/Research Project/tensorflow/images/emotions/sad/*.jpg'
path7 = 'C:/Users/Rushil/Desktop/BSC Computer Science/Honours/Research Project/tensorflow/images/emotions/surprise/*.jpg'

for file in glob(path1):
    angry.append(cv2.imread(file,0))

for file in glob(path2):
    disgust.append(cv2.imread(file,0))

for file in glob(path3):
    fear.append(cv2.imread(file,0))

for file in glob(path4):
    happy.append(cv2.imread(file,0))

for file in glob(path5):
    neutral.append(cv2.imread(file,0))

for file in glob(path6):
    sad.append(cv2.imread(file,0))

for file in glob(path7):
    surprise.append(cv2.imread(file,0))

angry = np.array(angry)
disgust = np.array(disgust)
fear = np.array(fear)
happy = np.array(happy)
neutral = np.array(neutral)
sad = np.array(sad)
surprise = np.array(surprise)

# Creating LBP of every image and then saving
os.chdir('C:/Users/Rushil/Desktop/BSC Computer Science/Honours/Research Project/tensorflow/images/new_ims_lbp/angry')
for i in range(len(angry)):
    temp = lbp(angry[i], 12,2)
    cv2.imwrite('angry'+str(i)+'.jpg', temp)

os.chdir('C:/Users/Rushil/Desktop/BSC Computer Science/Honours/Research Project/tensorflow/images/new_ims_lbp/disgust')
for i in range(len(disgust)):
    temp = lbp(disgust[i], 12,2)
    cv2.imwrite('disgust'+str(i)+'.jpg', temp)

os.chdir('C:/Users/Rushil/Desktop/BSC Computer Science/Honours/Research Project/tensorflow/images/new_ims_lbp/fear')
for i in range(len(fear)):
    temp = lbp(fear[i], 12,2)
    cv2.imwrite('fear'+str(i)+'.jpg', temp)

os.chdir('C:/Users/Rushil/Desktop/BSC Computer Science/Honours/Research Project/tensorflow/images/new_ims_lbp/happy')
for i in range(len(happy)):
    temp = lbp(happy[i], 12,2)
    cv2.imwrite('happy'+str(i)+'.jpg', temp)

os.chdir('C:/Users/Rushil/Desktop/BSC Computer Science/Honours/Research Project/tensorflow/images/new_ims_lbp/neutral')
for i in range(len(neutral)):
    temp = lbp(neutral[i], 12,2)
    cv2.imwrite('neutral'+str(i)+'.jpg', temp)

os.chdir('C:/Users/Rushil/Desktop/BSC Computer Science/Honours/Research Project/tensorflow/images/new_ims_lbp/sad')
for i in range(len(sad)):
    temp = lbp(sad[i], 12,2)
    cv2.imwrite('sad'+str(i)+'.jpg', temp)

os.chdir('C:/Users/Rushil/Desktop/BSC Computer Science/Honours/Research Project/tensorflow/images/new_ims_lbp/surprise')
for i in range(len(surprise)):
    temp = lbp(surprise[i], 12,2)
    cv2.imwrite('surprise'+str(i)+'.jpg', temp)