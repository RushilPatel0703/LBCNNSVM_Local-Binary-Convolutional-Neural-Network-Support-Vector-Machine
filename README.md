# LBCNNSVM_Local-Binary-Convolutional-Neural-Network-Support-Vector-Machine

Our problem domain is to classify one of the seven human expressions (anger, disgust, fear, happy, neutral, sad, surprise) given an image. Convolution Neural Networks (CNN) are very good at extracting robust information from images and generally perform well on their own. To accelerate the extraction process and point a CNN in the direction of the problem domain we can utilize Local Binary Patterns (LBP). Given our problem LBPs are ideal since they capture texture information. For classifications tasks in CNNs and other networks a softmax activation function is used for prediction and minimizing a cross entropy loss. We show that having a SVM layer that uses a Radial Basis Function (RBF) kernel instead of softmax gives us a slight advantage

## Dataset
https://www.kaggle.com/datasets/msambare/fer2013?select=train
