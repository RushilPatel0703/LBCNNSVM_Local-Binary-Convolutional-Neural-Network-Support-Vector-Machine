import matplotlib.pyplot as plt
import tensorflow as tf
from Models.CNNSVM import CNNSVM
import numpy as np

# Splitting data into training, validation and testing sets 
print('Getting Data...')
train = tf.keras.preprocessing.image_dataset_from_directory(
    'C:/Users/Rushil/Desktop/BSC Computer Science/Honours/Research Project/tensorflow/images/emotions',
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    batch_size=32,
    image_size=(48, 48),
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset='training')

validation = tf.keras.preprocessing.image_dataset_from_directory(
    'C:/Users/Rushil/Desktop/BSC Computer Science/Honours/Research Project/tensorflow/images/emotions',
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    batch_size=32,
    image_size=(48, 48),
    shuffle=True,
    seed=42,
    validation_split=0.2,
    subset='validation')

test = tf.keras.preprocessing.image_dataset_from_directory(
    'C:/Users/Rushil/Desktop/BSC Computer Science/Honours/Research Project/tensorflow/images/emotions',
    labels="inferred",
    label_mode="int",
    color_mode="grayscale",
    batch_size=32,
    image_size=(48, 48),
    shuffle=True,
    seed=42,
    validation_split=0.1,
    subset='training')

print('Building CNNSVM Model...')
model_cnnsvm = CNNSVM()
model_cnnsvm.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ['accuracy'])
cnnsvm_results = model_cnnsvm.fit(x = train, validation_data=validation, epochs = 100)


print('###################### ACCURACY ########################')
print(np.max(cnnsvm_results.history['accuracy']))

# Plots for Loss and Accuracy
plt.plot(cnnsvm_results.history['loss'], color ='#e71d36')
plt.title("CNNSVM Loss", fontsize=13)
plt.xlabel("Epochs", fontsize=11)
plt.ylabel("Loss", fontsize=11)
plt.show()
plt.savefig('CNNSVM_Loss_Plot')

plt.plot(cnnsvm_results.history['accuracy'], color ='#ff9f1c')
plt.title("CNNSVM Accuracy", fontsize=13)
plt.xlabel("Epochs", fontsize=11)
plt.xlabel("Accuracy", fontsize=11)
plt.show()
plt.savefig('CNNSVM_Accuracy_Plot')