from Models.CNN import CNN
import matplotlib.pyplot as plt
import tensorflow as tf
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
    validation_split=0.3,
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


print('Building CNN Model...')
model_cnn = CNN()
model_cnn.compile(optimizer="adam", loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics = ['accuracy'])
cnn_results = model_cnn.fit(x = train, validation_data=validation, epochs = 100)

print('###################### ACCURACY ########################')
print(np.max(cnn_results.history['accuracy']))

# Plots for Loss and Accuracy
plt.plot(cnn_results.history['loss'], color ='#e71d36')
plt.title("CNN Loss", fontsize=13)
plt.xlabel("Epochs", fontsize=11)
plt.ylabel("Loss", fontsize=11)
plt.show()
plt.savefig('CNN_Loss_Plot')

plt.plot(cnn_results.history['accuracy'], color ='#ff9f1c')
plt.title("CNN Accuracy", fontsize=13)
plt.xlabel("Epochs", fontsize=11)
plt.xlabel("Accuracy", fontsize=11)
plt.show()
plt.savefig('CNN_Accuracy_Plot')