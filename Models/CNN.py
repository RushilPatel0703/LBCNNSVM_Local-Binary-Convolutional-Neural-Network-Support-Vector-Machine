import tensorflow as tf
from keras.models import Sequential

class CNN(tf.keras.Model):
  def __init__(self):
    super(CNN, self).__init__()

    self.norm = tf.keras.layers.experimental.preprocessing.Rescaling(1./255) # normalising the input
    
    #first convolution layer
    self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
    self.max1 = tf.keras.layers.MaxPooling2D()
    
    #second convolution layer
    self.conv2 = tf.keras.layers.Conv2D(32, 3, activation='relu')
    self.max2 = tf.keras.layers.MaxPooling2D()
    
    #third convolution layer
    self.conv3 = tf.keras.layers.Conv2D(32, 3, activation='relu')
    self.max3 = tf.keras.layers.MaxPooling2D()

    #Fully connected layer
    self.flat = tf.keras.layers.Flatten()
    self.dense1 = tf.keras.layers.Dense(128, activation='softmax')
    self.dropout = tf.keras.layers.Dropout(0.3)
    
    self.dense2 = tf.keras.layers.Dense(7)

  def call(self, x):
    x = self.norm(x)

    x = self.conv1(x)
    x = self.max1(x)

    x = self.conv2(x)
    x = self.max2(x)

    x = self.conv3(x)
    x = self.max3(x)

    x = self.flat(x)

    x = self.dense1(x)
    x = self.dropout(x)
    x = self.dense2(x)

    return x
