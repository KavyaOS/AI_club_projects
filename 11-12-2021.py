from tensorflow import keras
import numpy as np
from matplotlib import pyplot as plt

layer = keras.layers.Dense(3, input_shape = (3,), activation = 'relu') #relu trains faster than sigmoid does

ex = np.array([[1,2,3]])

#print(layer(ex).numpy())

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# plt.imshow(x_train[0])
# plt.show()

x_train = x_train / 255  #Normalize values
#print(x_train[0])

x_train = x_train.reshape(60000, 28*28) #Flattened our training data to be one dimensional
#print(x_train) #

model = keras.models.Sequential()
model.add(keras.layers.Dense(10, input_shape = (28*28, ), activation = 'relu'))
model.add(keras.layers.Dense(10, activation = 'relu'))

model.add(keras.layers.Dense(10, activation = 'softmax'))

model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs = 3)

model.predict(x_train[:4])

# LET'S OBSERVE THE MODEL OURSELF
# Change the value of 'INDEX' to check differnt example
# Values between 0 and 9999 will work
INDEX = 24


print('Predicted:', model.predict(x_test[INDEX:INDEX+1].reshape((-1,784))), '\n But that is kind of hard to read...')

print('\n\n\nPredicted:', model.predict(x_test[INDEX:INDEX+1].reshape((-1,784))).argmax(), '\n The numpy array function ".argmax()" returns the index with the highest value')

plt.imshow(x_test[INDEX])
plt.show()