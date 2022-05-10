import numpy as np
import tensorflow as tf
from keras.layers import Dense

import matplotlib.pyplot as plt

def flood_categorical_crossentropy(y_true, y_pred):
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    loss = tf.math.abs(loss - b) + b
    return loss

#testing flooding for MNIST

#import the dataset

print("Testing benefits of Flooding for MLP trained on MNIST dataset\n")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print("Shape of x_train: {} y_train: {}".format(x_train.shape, y_train.shape))
print("Shape of x_test: {} y_test: {}".format(x_test.shape, y_test.shape))


#data preprocessing
x_train = x_train.reshape(60000, 784) #reshape 28 x 28 image to 784-length vectors.
x_test = x_test.reshape(10000, 784)   #reshape 28 x 28 image to 784-length vectors.

x_train = x_train.astype("float32")   #change int to float
x_test = x_test.astype("float32")

x_train /= 255                        #normalizing
x_test /= 255


num_classes = 1000                           #number of classes
y_train = tf.one_hot(y_train, num_classes)   #one hot encoding
y_test = tf.one_hot(y_test, num_classes)

print("Training matrix shape", x_train.shape)
print("Testing matrix shape", x_test.shape)


#define the MLP model
num_nodes = 1000                      #number of nodes in hidden layers
batch_size = 200

model = tf.keras.Sequential()
model.add(Dense(num_nodes, input_shape=(784,), activation="relu"))
model.add(Dense(num_nodes, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))


#set flood value
b = 0.01

SGD = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9,)
model.compile(loss=flood_categorical_crossentropy, optimizer=SGD, metrics=["mse", "acc"])
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=10, validation_data=(x_test, y_test), verbose=0)
model.evaluate(x_test,  y_test, verbose=2)

#plot figure
fig = plt.figure()
plt.plot(history.history['loss'], label="Train loss")
plt.plot(history.history['val_loss'], label="Test loss")
plt.legend()
plt.show()

#define the MLP model1
model1 = tf.keras.Sequential()
model1.add(Dense(num_nodes, input_shape=(784,), activation="relu"))
model1.add(Dense(num_nodes, activation="relu"))
model1.add(Dense(num_classes, activation="softmax"))

SGD = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9,)
model1.compile(loss="categorical_crossentropy", optimizer=SGD, metrics=["mse", "acc"])
history1 = model1.fit(x_train, y_train, batch_size=batch_size, epochs=10, validation_data=(x_test, y_test), verbose=0)
model1.evaluate(x_test,  y_test, verbose=2)

fig = plt.figure()
plt.plot(history1.history['loss'], label="Train loss")
plt.plot(history1.history['val_loss'], label="Test loss")
plt.legend()
plt.show()

print("Testing Loss: ", history1.history['val_loss'][-1])
print("Testing Loss w/ Flooding: ", history.history['val_loss'][-1])
