# -*- coding: utf-8 -*-
"""
Created on Sun Apr  4 11:34:16 2021

@author: HSingh
"""


#Preprocessing

#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading the dataset

df=pd.read_csv('Churn_Modelling.csv')
X = df.iloc[:, 3:13]
y = df.iloc[:, 13]

#creating dummy for object data
#dummy is not created for first to save memory
geog=pd.get_dummies(X["Geography"],drop_first=True)
sex=pd.get_dummies(X['Gender'],drop_first=True)

#concatenating dummy and dropping original var
X=pd.concat([X,geog,sex],axis=1)
X=X.drop(['Geography','Gender'],axis=1)

#preparing the training and test data
from sklearn.model_selection import train_test_split
x_tr,x_ts,y_tr,y_ts=train_test_split(X,y,test_size=0.3)

#Scaling the features
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
x_tr=ss.fit_transform(x_tr)
x_ts=ss.fit_transform(x_ts)

#preparing the ANN

#importing libraries
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout

#initializing the classifier
cl=Sequential()

#Adding first hidden layer
cl.add(Dense(units=6,kernel_initializer='he_uniform',activation='relu',input_dim=11))

#second hidden layer
cl.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu'))

#output layer
cl.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))

#compiling the classifier
cl.compile(optimizer='Adamax', loss='binary_crossentropy',metrics=['accuracy'])

#Fitting the training data
model=cl.fit(x_tr, y_tr,validation_split=0.33, batch_size = 10, epochs = 100)

# list all data in history

print(model.history.keys())

#Plotting the accuracy and loss values with epochs
plt.plot(model.history['accuracy'])
plt.plot(model.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()

#Doing the predictions
y_pred=cl.predict(x_ts)
y_pred=(y_pred>0.5)

#Evaluating the performance matrices
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print(accuracy_score(y_ts,y_pred))
print(confusion_matrix(y_ts,y_pred))



