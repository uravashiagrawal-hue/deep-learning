import numpy as np
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
	for filename in filenames:
		print(os.path.join(dirname, filename))

df = pd.read_csv('MUlti layer perceptron\Churn_Modelling.csv')
print(df.head())

df.drop(columns = ['Surname', 'CustomerId', 'RowNumber'], inplace = True)
print(df.sample())

# converting categorical column to numerical by encoding
df = pd.get_dummies(df,columns =['Geography','Gender'], drop_first = True)
print(df.head())

x= df.drop(columns=['Exited'])
y= df['Exited'].values

from sklearn.model_selection import train_test_split
x_train,x_test, y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

x_train_trf = scaler.fit_transform(x_train)
x_test_trf = scaler.transform(x_test)

import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(11, activation = 'sigmoid', input_dim = 11))
model.add(Dense(11, activation = 'sigmoid'))
model.add(Dense(1, activation = 'sigmoid'))

model.summary()

model.compile(optimizer='Adam',loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit(x_train,y_train,batch_size=50,epochs=100,verbose=1,validation_split=0.2)

y_pred = model.predict(x_test)
print(y_pred)

y_pred = y_pred.argmax(axis = -1)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()
