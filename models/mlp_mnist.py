from gc import callbacks
import numpy as np
import tensorflow as tf
from keras.layers import Dense
import matplotlib.pyplot as plt
import datetime
import os

# Precision, Recall and f1 score
from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

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

model = tf.keras.Sequential()
model.add(Dense(num_nodes, input_shape=(784,), activation="relu"))
model.add(Dense(num_nodes, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))


#set flood value
b = 0.01                              #selecting value of b from {0.01 .. 0.1}

#for categorical crossentropy
def flood_categorical_crossentropy(y_true, y_pred):
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    loss = tf.math.abs(loss - b) + b
    return loss

#add support for tensorboard
log_dir="logs/w_flood_mnist/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

SGD = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9,)
model.compile(loss=flood_categorical_crossentropy, optimizer=SGD, metrics=["mse", "acc"])
#model.compile(loss=flood_categorical_crossentropy, optimizer=SGD, metrics=["acc","mse", precision_m,recall_m,f1_m])
history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test), callbacks=[tensorboard_callback])
model.evaluate(x_test,  y_test, verbose=2)

#plot loss
'''
fig = plt.figure()
plt.plot(history.history['loss'], label="Train loss")
plt.plot(history.history['val_loss'], label="Test loss")
plt.legend()
plt.show()
'''

#define the MLP model1
model1 = tf.keras.Sequential()
model1.add(Dense(num_nodes, input_shape=(784,), activation="relu"))
model1.add(Dense(num_nodes, activation="relu"))
model1.add(Dense(num_classes, activation="softmax"))

#add support for tensorboard
log_dir_1="logs/wo_flood_mnist/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback_1 = tf.keras.callbacks.TensorBoard(log_dir=log_dir_1, histogram_freq=1)

SGD = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9,)
model1.compile(loss="categorical_crossentropy", optimizer=SGD, metrics=["mse", "acc"])  #using categorical_crossentropy loss
#model1.compile(loss= "flood_categorical_crossentropy", optimizer=SGD, metrics=["acc","mse", precision_m,recall_m,f1_m])
history1 = model1.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test),callbacks=[tensorboard_callback_1])
model1.evaluate(x_test,  y_test, verbose=2)

#plot loss
'''
fig = plt.figure()
plt.plot(history1.history['loss'], label="Train loss")
plt.plot(history1.history['val_loss'], label="Test loss")
plt.legend()
plt.show()
'''

#comparing accuracies
fig = plt.figure()
plt.plot(history1.history['acc'], label="Training acc")
plt.plot(history.history['acc'], label="Training acc w/ flooding")
plt.plot(history1.history['val_acc'], label="Testing acc")
plt.plot(history.history['val_acc'], label='Tesiting acc w/ flooding')
plt.legend()
plt.show()

#comparing losses
fig = plt.figure()
plt.plot(history1.history['loss'], label="Training loss")
plt.plot(history.history['loss'], label="Training loss w/ flooding")
plt.plot(history1.history['val_loss'], label="Testing loss")
plt.plot(history.history['val_loss'], label='Testing loss w/ flooding')
plt.legend()
plt.show()
