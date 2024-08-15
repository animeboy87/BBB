# Practical 3
# Implementing a deep neural network for performing binary classification tasks

import pandas as pd
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

dataset=pd.read_csv('/content/diabetes.csv',delimiter=',')
dataset


X=dataset.iloc[:,0:8]
y=dataset.iloc[:,8]
X
y


model=Sequential()
model.add(Dense(12,input_dim=8,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam' ,metrics=['accuracy'])
model.fit(X,y,epochs=150,batch_size=10)


_,accuracy=model.evaluate(X,y)
print('Accuracy of model is',(accuracy*100))


