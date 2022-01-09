#!/usr/bin/env python3
#libraries
#Train the B-VAE network with the selected n,B hyperparameters
#input: Training dataset train.csv
#output: Saved Encoder network weights, encoder-deconder weights (we do not use this in our work), reconstruction folder with images
import time
import random
import csv
import cv2
import os
import glob
import numpy as np
import keras
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam, SGD
from keras.models import model_from_json, load_model
from keras.layers import Input, Dense
from keras.models import Model,Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Convolution2D as Conv2D
from keras.layers.convolutional import Deconv2D as Conv2DTranspose
from keras.layers import Lambda, Input, Dense, MaxPooling2D, BatchNormalization,Input
from keras.layers import UpSampling2D, Dropout, Flatten, Reshape, RepeatVector, LeakyReLU,Activation
from keras.callbacks import ModelCheckpoint
from keras.losses import mse, binary_crossentropy
from keras.callbacks import EarlyStopping, LearningRateScheduler
seed = 7
np.random.seed(seed)
from keras.callbacks import CSVLogger
from keras.callbacks import History
from itertools import product
import matplotlib.pyplot as plt
import prettytable
from prettytable import PrettyTable
from sklearn.utils import shuffle
from glob import glob

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"#Setting the script to run on GPU:1 (remove this if you do not have a gpu)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'



#Load complete input images without shuffling
def load_training_images(trainroot): # New: whole function
    # inputs = []
    # with open(train_data + 'train.csv', 'rt') as csvfile:
            # reader = csv.reader(csvfile)
            # for row in reader:
                # img = cv2.imread(train_data + row[0])
                # img = cv2.resize(img, (224, 224))
                # img = img / 255.
                # inputs.append(img)
    train_folders = [6, 20, 17, 7, 30, 8, 13, 27, 5, 26, 31, 21, 32, 3, 10, 19, 1, 24, 4, 2]
    locs = []
    for folder_number in train_folders:
        if folder_number <= 10:
            locs.append(trainroot+"setting_1/"+str(folder_number))
        elif folder_number >= 11 and folder_number <= 21:
            locs.append(trainroot+"setting_2/"+str(folder_number-11))
        elif folder_number >= 22 and folder_number <= 32:
            locs.append(trainroot+"setting_3/"+str(folder_number-22))
    
    inputs = []
    for idx, scenefolder in enumerate(locs):
        for imagefile in sorted(glob(scenefolder + "/*.png")):
            img = cv2.imread(imagefile)
            img = cv2.resize(img, (224, 224))
            img = img / 255.
            inputs.append(img)
			
    print("Total number of images:%d" %len(inputs))
    return inputs

#Reshape input images
def data_reshape(input_image):
    img_train, img_test = np.array(input_image[0:len(input_image)-300].copy()), np.array(input_image[len(input_image)-300:len(input_image)].copy())
    img_train = np.reshape(img_train, [-1, img_train.shape[1],img_train.shape[2],img_train.shape[3]])
    img_test = np.reshape(img_test, [-1, img_test.shape[1],img_test.shape[2],img_test.shape[3]])
    inp = (img_train, img_test)
    return inp

#Create the Beta-VAE model
def CreateModels(nl, b):
    #sampling function of the Beta-VAE
    def sample_func(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean = 0 and std = 1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    model = Sequential()
    input_img = Input(shape=(224,224,3), name='image')
    x = Conv2D(128, (3, 3),  use_bias=False, padding='same')(input_img)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), padding='same',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), padding='same',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), padding='same',use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(2048)(x)
    x = LeakyReLU(0.1)(x)
    x = Dense(1000)(x)
    x = LeakyReLU(0.1)(x)
    x = Dense(250)(x)
    x = LeakyReLU(0.1)(x)

    z_mean = Dense(nl, name='z_mean')(x)
    z_log_var = Dense(nl, name='z_log_var')(x)
    z = Lambda(sample_func, output_shape=(nl,), name='z')([z_mean, z_log_var])
    encoder = Model(input_img, [z_mean, z_log_var, z], name='encoder')
    #encoder.summary()

    latent_inputs = Input(shape=(nl,), name='z_sampling')

    x = Dense(250)(latent_inputs)
    x = LeakyReLU(0.1)(x)
    x = Dense(1000)(x)
    x = LeakyReLU(0.1)(x)
    x = Dense(2048)(x)
    x = LeakyReLU(0.1)(x)
    x = Dense(3136)(x)
    x = LeakyReLU(0.1)(x)
    x = Reshape((14, 14, 16))(x)
    x = Conv2D(16, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(32, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(64, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(128, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.1)(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(3, (3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    decoded = Activation('sigmoid')(x)

    decoder = Model(latent_inputs, decoded)
    outputs = decoder(encoder(input_img)[2])
    autoencoder = Model(input_img,outputs)
    #autoencoder.summary()

    #define custom loss function of the Beta-VAE
    def vae_loss(true, pred):
        rec_loss = mse(K.flatten(true), K.flatten(pred))
        rec_loss *= 224*224
        KL_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        KL_loss = K.sum(KL_loss, axis=-1)
        KL_loss *= -0.5
        vae_loss = K.mean(rec_loss + b*(KL_loss))
        return vae_loss

    def lr_scheduler(epoch): #learningrate scheduler to adjust learning rate.
        lr = 1e-4
        if epoch > 100:
            #print("New learning rate")
            lr = 1e-5
        if(epoch > 125):
             lr = 1e-6
        return lr

    scheduler = LearningRateScheduler(lr_scheduler)

    #Define adam optimizer
    adam = keras.optimizers.Adam(lr=0.0000001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False) 
    autoencoder.compile(optimizer='adam',loss=vae_loss, metrics=[vae_loss])

    return autoencoder, encoder, z_log_var,scheduler

#Train and save the B-VAE models
def train_w_s(X,autoencoder,epoch_number,batch_size_number,path,dir_path,scheduler):
    X_train, X_test = X
    data = CSVLogger(dir_path + '/loss.csv', append=True, separator=';')
    filePath = dir_path + '/weights.best.hdf5'#checkpoint weights
    checkpoint = ModelCheckpoint(filePath, monitor='vae_loss', verbose=1, save_best_only=True, mode='min')
    EarlyStopping(monitor='vae_loss', patience=40, verbose=0),
    callbacks_list = [checkpoint, data]
    autoencoder.fit(X_train, X_train, epochs=epoch_number, batch_size=batch_size_number, shuffle=True, validation_data=(X_test, X_test), callbacks=callbacks_list, verbose=2)

#Save the autoencoder model
def SaveAutoencoderModel(autoencoder,dir_path):
	auto_model_json = autoencoder.to_json()
	with open(dir_path + '/auto_model.json', "w") as json_file:
		json_file.write(auto_model_json)
	autoencoder.save_weights(dir_path + '/auto_model.h5')
	print("Saved Autoencoder model to disk")

#Save the encoder model
def SaveEncoderModel(encoder,dir_path):
	en_model_json = encoder.to_json()
	with open(dir_path + '/en_model.json', "w") as json_file:
		json_file.write(en_model_json)
	encoder.save_weights(dir_path + '/en_model.h5')
	print("Saved Encoder model to disk")

#Test the trained models on a different test data
def test(autoencoder,encoder,test):
    autoencoder_res = autoencoder.predict(test)
    encoder_res = encoder.predict(test)
    res_x = test.copy()
    res_y = autoencoder_res.copy()
    res_x = res_x * 255
    res_y = res_y * 255

    return res_x, res_y, encoder_res

#Save the reconstructed test data in a separate folder.
#For this create a folder named results in the directory you are working in.
def savedata(test_in, test_out, test_encoded, Working_path):
    trainfolder = Working_path + "reconstruction" + '/'
    os.makedirs(trainfolder, exist_ok=True)
    for i in range(len(test_in)):
        test_in = np.reshape(test_in,[-1, 224,224,3])#Reshape the data
        test_out = np.reshape(test_out,[-1, 224,224,3])#Reshape the data
        cv2.imwrite(trainfolder + str(i) +'_in.png', test_in[i])
        cv2.imwrite(trainfolder + str(i) +'_out.png', test_out[i])


if __name__ == '__main__':
    train_data = '../../carla_data/training/' # "/home/scope/Carla/CARLA_0.9.6/PythonAPI/TCPS-data/Train-data/"
    input_image = load_training_images(train_data)
    input_image = shuffle(input_image)
    inp = data_reshape(input_image)
    Latents = 30 #hyperparameter1
    betas = [1.4] #,1.3,1.4,1.5,2.0,3.0,4.0,5.0] #hyperparameter2
    epoch_number= 90 # ORIG: 150 but got NAN losses after 92 epochs #epoch numbers for hyperparameter tuning and training
    batch_size_number=16 #batch size for hyperparameter tuning and training
    path = "" # "/home/scope/Carla/CARLA_0.9.6/PythonAPI/SVDD/"
    print("******************************Hyperparameter Tuning******************************************")
    #rand_x, rand_y=hyperparameter_search(trial,iterations,Latents,betas,inp,epoch_number,batch_size_number)# call the hyperparameter tuning function
    print("******************************Training******************************************")
    for i in range(len(betas)):
        dir_path = "../carla_models/"
        data_store = dir_path + '%d_%0.1f'%(Latents,betas[i]) + '/'
        os.makedirs(data_store, exist_ok=True)
        autoencoder,encoder,z_log_var,scheduler = CreateModels(Latents,betas[i])# Running the autoencoder model
        train_w_s(inp,autoencoder,epoch_number,batch_size_number,path,data_store,scheduler)#Train the selected B-VAE parameters and train the models and save them.
        SaveAutoencoderModel(autoencoder,data_store)#Save full autoencoder model
        SaveEncoderModel(encoder,data_store)#Save encoder model
        test_in, test_out, test_encoded = test(autoencoder,encoder,inp[1])#Test the autoencoder model with training weights.
        savedata(test_in, test_out, test_encoded, data_store)#Save the data
