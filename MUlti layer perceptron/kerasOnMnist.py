import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten

(x_train,y_train),(x_test,y_test) = keras.datasets.mnist.load_data()
print(x_test.shape)
print(y_train)

import matplotlib.pyplot as plt
plt.imshow(x_train[2])
plt.show()

print(x_train[0])
# we will arrage all the values in same range 0-1
x_train = x_train/255
x_test = x_test /255
print(x_train[0])

model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'Adam',metrics = ['accuracy'])
history = model.fit(x_train,y_train,epochs = 25, validation_split = 0.2)

y_prob = model.predict(x_test)
y_pred = y_prob.argmax(axis =1)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()

print(model.predict(x_test[1].reshape(1,28,28)).argmax(axis=1))
