import tensorflow as tf


class CNNSVM(tf.keras.Model):
  def __init__(self):
    super(CNNSVM, self).__init__()

    self.norm = tf.keras.layers.experimental.preprocessing.Rescaling(1./255) #normalisation of data
  
    #first convolution layer
    self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
    self.max1 = tf.keras.layers.MaxPooling2D()
    self.drop1 = tf.keras.layers.Dropout(0.2)
    
    #second convolution layer
    self.conv2 = tf.keras.layers.Conv2D(32, 3, activation='relu')
    self.max2 = tf.keras.layers.MaxPooling2D()
    self.drop2 = tf.keras.layers.Dropout(0.2)

    #third convolution layer
    self.conv3 = tf.keras.layers.Conv2D(32, 3, activation='relu')
    self.max3 = tf.keras.layers.MaxPooling2D()
    self.drop3 = tf.keras.layers.Dropout(0.2)

    self.flat = tf.keras.layers.Flatten()
    
    #fully connected layer
    self.dense1 = tf.keras.layers.Dense(256, activation='relu')
    self.dense2 = tf.keras.layers.Dense(256, activation='relu')
    self.drop4 = tf.keras.layers.Dropout(0.2)
    self.dense3 = tf.keras.layers.Dense(128, activation='relu')
    
    # SVM layer
    self.svm = tf.keras.layers.experimental.RandomFourierFeatures(output_dim=7, kernel_initializer='gaussian', trainable=True)#using Gaussian Radial Basis Function

  def call(self, x):
    x = self.norm(x)

    x = self.conv1(x)
    x = self.max1(x)
    x = self.drop1(x)

    x = self.conv2(x)
    x = self.max2(x)
    x = self.drop2(x)

    x = self.conv3(x)
    x = self.max3(x)
    x = self.drop3(x)

    x = self.flat(x)

    x = self.dense1(x)
    x = self.dense2(x)
    x = self.drop4(x)
    x = self.dense3(x)

    x = self.svm(x)

    return x
