#!/usr/bin/env python3
#libraries
import keras
import tensorflow as tf
tf.executing_eagerly()
from keras import backend as K
import cv2
import os
import sys
import csv
import glob
import numpy as np
import time
from sklearn.utils import shuffle
from keras.optimizers import Adam,rmsprop,SGD
from keras.models import model_from_json, load_model
from keras.layers import Input, Dense
from keras.models import Model,Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Convolution2D as Conv2D
from keras.layers.convolutional import Deconv2D as Conv2DTranspose
from keras.layers import Lambda, Input, Dense, MaxPooling2D, BatchNormalization,Input
from keras.layers import UpSampling2D, Dropout, Flatten, Reshape, RepeatVector, LeakyReLU,Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.losses import mse, binary_crossentropy
keras.callbacks.TerminateOnNaN()
seed = 7
np.random.seed(seed)
from keras.callbacks import CSVLogger

os.environ["CUDA_VISIBLE_DEVICES"]="1"#Setting the script to run on GPU:1,2

#Load complete input images without shuffling
def load_images(path, folders):
    numImages = 0
    inputs = []
    for folder in folders:
        paths = path + folder + '/'
        print(paths)
        numFiles = len(glob.glob1(paths,'*.png'))
        numImages += numFiles
        for img in glob.glob(paths+'*.png'):
            img = cv2.imread(img)
            img = cv2.resize(img, (200,66))
            img = img /255.
            inputs.append(img)
        #inpu = shuffle(inputs)
    print("Total number of images:%d" %len(inputs))
    return inputs


def load_steering_value(path, folders):
    miny= -1
    maxy= 1
    Y=[]
    for folder in folders:
        dataset = path + folder + '/steer.csv'
        with open(dataset, 'rt') as csvfile:
              reader = csv.reader(csvfile)
              for row in reader:
                output=[]
                x=(float(row[0])-miny)/(maxy-miny)
                output.append(x)
                Y.append(output)
    print("Total Steering values:%d"%len(Y))
    return Y

def createModel():
    model = Sequential([
        Conv2D(24, kernel_size=(5,5), strides=(2,2), activation='relu', input_shape=(66,200,3)),
        BatchNormalization(),
        Conv2D(36, kernel_size=(5,5), strides=(2,2), activation='relu'),
        BatchNormalization(),
        Conv2D(48, kernel_size=(5,5), strides=(2,2), activation='relu'),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3,3), strides=(1,1), activation='relu'),
        BatchNormalization(),
        Conv2D(64, kernel_size=(3,3), strides=(1,1), activation='relu'),
        BatchNormalization(),
        Flatten(),
        Dense(100, activation='relu'),
        BatchNormalization(),
        Dense(50, activation='relu'),
        BatchNormalization(),
        Dense(10, activation='relu'),
        BatchNormalization(),
        Dense(1)
    ])
    return model


def trainModel(model, X, Y):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05,random_state=42)
    adam = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss="mse", metrics=['mae', 'acc'])
    filePath = save_path + "weights.best.hdf5"
    callbacks_list = [
    ModelCheckpoint(filePath, monitor='val_mean_absolute_error', verbose=1, save_best_only=False),
    EarlyStopping(monitor='val_mean_absolute_error', patience=10, verbose=0)]
    model.fit(X_train, Y_train, epochs=20, batch_size=64,validation_data=(X_test, Y_test),callbacks=callbacks_list, verbose=2)


def saveModel(model,save_path):
	model_json = model.to_json()
	with open(save_path + "model.json", "w") as json_file:
		json_file.write(model_json)

	model.save_weights(save_path + "model.h5")
	print("Saved model to disk")


if __name__ == '__main__':
    data_path = "/home/scope/Carla/B-VAE-OOD-Monitor/data-generation/results/"
    save_path = "/home/scope/Carla/B-VAE-OOD-Monitor/LEC/results/"
    trainingFolders = ["scene0"]
    image_input = load_images(data_path,trainingFolders)
    steer_output = load_steering_value(data_path,trainingFolders)
    image_input = np.array(image_input)
    steer_output = np.array(steer_output)
    print(image_input.shape)
    print(steer_output.shape)
    model = createModel()
    print("create a new model")
    trainModel(model, image_input, steer_output)
    print("completed training the model")
    saveModel(model,save_path)
