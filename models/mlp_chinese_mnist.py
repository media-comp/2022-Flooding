import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import random
import cv2 as cv2
import pandas as pd
import tensorflow as tf
from keras.layers import Dense
import matplotlib.pyplot as plt
import datetime

def load_images_from_folder(folder):
    images = np.zeros((15000,64,64))
    np.asarray(images,dtype="uint8")
    i=0
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images[i,:,:]=gray
        i=i+1
    return images

#please download the chinese minst dataset and change the images and dataframe directory for training

images=load_images_from_folder(r"C:\Users\82583\Downloads\2022-Flooding-main\data\archive\data\data")

x_train = images.reshape(15000, 4096)

x_train=np.asarray(x_train,dtype="float32")

dataframe = pd.read_csv(r"C:\Users\82583\Downloads\2022-Flooding-main\data\archive\chinese_mnist.csv", low_memory = False)
code=np.array(dataframe.code)

x_train/=255

num_classes = 15                           #number of classes
y_train = tf.one_hot(code, num_classes)   #one hot encoding

num_nodes = 1000                      #number of nodes in hidden layers

model = tf.keras.Sequential()
model.add(Dense(num_nodes, input_shape=(4096,), activation="relu"))
model.add(Dense(num_nodes, activation="relu"))
model.add(Dense(15, activation="softmax"))


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

SGD = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
model.compile(loss=flood_categorical_crossentropy,optimizer=SGD, metrics=["mse", "acc"])
history = model.fit(x_train, y_train, epochs=100,callbacks=[tensorboard_callback],batch_size=1000)


