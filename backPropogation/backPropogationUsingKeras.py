import numpy as np
import pandas as pd
df = pd.DataFrame([[8,8,4],[7,9,5],[6,10,6],[5,12,7]], columns =['cgpa','profit_score','lpa'])
print(df)

import tensorflow
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense

model = Sequential()
model.add(Dense(2, activation='linear', input_dim = 2))
model.add(Dense(1, activation='linear'))

model.summary()

# keras initialise weight randomly
model.get_weights()

# creating new weights
new_weights = [np.array([[0.1,0.1],[0.1,0.1]], dtype = np.float32),
			   np.array([0.,0.],dtype = np.float32),
			   np.array([[0.1],[0.1]], dtype = np.float32),
			   np.array([0.], dtype=np.float32)]

# setting these new weight
model.set_weights(new_weights)

model.get_weights()

optimizer = keras.optimizers.Adam(learning_rate = 0.001)
model.compile(loss='mean_squared_error', optimizer = optimizer)

model.fit(df.iloc[:,0:-1].values, df['lpa'].values,epochs = 75, verbose =1, batch_size =1)

