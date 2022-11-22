# LBCNNSVM_Local-Binary-Convolutional-Neural-Network-Support-Vector-Machine

Our problem domain is to classify one of the seven human expressions (anger, disgust, fear, happy, neutral, sad, surprise) given an image. Convolution Neural Networks (CNN) are very good at extracting robust information from images and generally perform well on their own. To accelerate the extraction process and point a CNN in the direction of the problem domain we can utilize Local Binary Patterns (LBP). Given our problem LBPs are ideal since they capture texture information. For classifications tasks in CNNs and other networks a softmax activation function is used for prediction and minimizing a cross entropy loss. We show that having a SVM layer that uses a Radial Basis Function (RBF) kernel instead of softmax gives us a slight advantage

## Dataset
https://www.kaggle.com/datasets/msambare/fer2013?select=train

## Results 

<p float="left">
  <img src="https://user-images.githubusercontent.com/30756824/203409961-a76ef833-0089-4f56-acbe-932da536d35c.jpg" width="400" />
  <img src="https://user-images.githubusercontent.com/30756824/203410040-813d8b7a-31a6-4da5-a850-5eb9cac06a29.jpg" width="400" /> 
</p>

