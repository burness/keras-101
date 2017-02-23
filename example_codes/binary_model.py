from keras.models import Sequential
from keras.layers import Dense

# it is just for being fimilar with Keras
model = Sequential()
model.add(Dense(1, input_dim=784, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

import numpy as np
from sklearn.model_selection import train_test_split

data = np.random.random((1300, 784))
labels = np.random.randint(2, size=(1300, 1))

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.33, random_state=42)

model.fit(train_data, train_labels, nb_epoch=10, batch_size=32)
score = model.evaluate(test_data, test_labels, batch_size=16)
print score