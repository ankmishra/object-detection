#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 23:14:56 2018

@author: ankit
"""

from keras.datasets import cifar10
from keras. models import Sequential  
from keras. layers import Dense  
from keras. layers import Dropout  
from keras. layers import Flatten  
import numpy as np
from matplotlib import pyplot as plt
from keras. layers . convolutional import Conv2D  
from keras. layers . convolutional import MaxPooling2D  
from keras. utils import np_utils 
# import cv2
import matplotlib. pyplot as plt 


( X_train , y_train ) , ( X_test , y_test ) = cifar10.load_data ( )

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255
X_test = X_test / 255
y_train = np_utils. to_categorical ( y_train )
y_test = np_utils. to_categorical ( y_test )
num_classes = y_test. shape [ 1 ]

model = Sequential ( )

model.add ( Conv2D ( 32 , ( 5 , 5 ) , input_shape = (32,32,3) , activation = 'relu' ) )

model.add ( MaxPooling2D ( pool_size = ( 2 , 2 ) ) )

model.add ( Conv2D ( 32 , ( 3 , 3 ) , activation = 'relu' ) )

model.add ( MaxPooling2D ( pool_size = ( 2 , 2 ) ) )

model.add ( Dropout ( 0.2 ) )

model.add ( Flatten ( ) )

model.add ( Dense ( 128 , activation = 'relu' ) )  

model.add ( Dense ( num_classes, activation = 'softmax' , name = 'preds' ) )

model.compile ( loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = [ 'accuracy' ] )

model.summary ( )
model.fit ( X_train , y_train, validation_data = ( X_test , y_test ) , epochs = 20 , batch_size = 32 )
scores = model.evaluate ( X_test , y_test, verbose = 0 )
print ( "\ nacc:% .2f %%" % (scores [1] * 100)) 

# img_pred = cv2.imread ('deer.jpg', 0)
# img_pred.shape
# # forces the image to have the input dimensions equal to those used in the training data (28x28)
# if img_pred.shape!= [32,32]:
#     img2 = cv2.resize (img_pred, (32,96))
#     img_pred = img2.reshape ((32,32,3))
# else:
#     img_pred = img_pred.reshape ((32,32,3))
    

# # here also we inform the value for the depth = 1, number of rows and columns, which correspond 28x28 of the image.
# img_pred = img_pred.reshape (1,32,32,3)

# pred = model.predict_classes (img_pred)

# pred_proba = model.predict_proba (img_pred)
# pred_proba = "% .2f %%"% (pred_proba [0] [pred] * 100)

# print (pred [0], "with probability of", pred_proba)