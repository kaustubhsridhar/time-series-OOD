#!/usr/bin/env python3
#libraries
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
from keras.optimizers import Adam,SGD
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
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
from keras.callbacks import History
from itertools import product
import matplotlib.pyplot as plt
import prettytable
from prettytable import PrettyTable
from sklearn.utils import shuffle
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler
from keras.callbacks import Callback, LearningRateScheduler
import warnings
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean,median
from keras.callbacks import TerminateOnNaN

plt.style.use('ggplot')
warnings.filterwarnings("ignore")

import resource

class MemoryCallback(Callback):
    def on_epoch_end(self, epoch, log={}):
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


os.environ["CUDA_VISIBLE_DEVICES"]="3"#Setting the script to run on GPU:1,2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

class new_callback(tf.keras.callbacks.Callback):
    def epoch_end(epoch, logs={}):
        if(logs.get('val_loss') == nan): # select the accuracy
            print("\n !!! no further training !!!")
            model.stop_training = True


#Load complete input images without shuffling
def load_training_images(train_data):
    inputs = []
    comp_inp = []
    with open(train_data + 'train.csv', 'rt') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                img = cv2.imread(train_data + row[0])
                img = cv2.resize(img, (224, 224))
                img = img / 255.
                inputs.append(img)
            for i in range(0,len(inputs),3):
                    comp_inp.append(inputs[i])
            print("Total number of images:%d" %len(comp_inp))
            return inputs, comp_inp

#Reshape input images
def data_reshape(input_image):
    img_train, img_test = np.array(input_image[0:len(input_image)-200].copy()), np.array(input_image[len(input_image)-200:len(input_image)].copy())
    img_train = np.reshape(img_train, [-1, img_train.shape[1],img_train.shape[2],img_train.shape[3]])
    img_test = np.reshape(img_test, [-1, img_test.shape[1],img_test.shape[2],img_test.shape[3]])
    inp = (img_train, img_test)
    return inp

class new_callback(tf.keras.callbacks.Callback):
    def epoch_end(epoch, logs={}):
        if(logs.get('val_loss') == nan): # select the accuracy
            print("\n !!! no further training !!!")
            model.stop_training = True

#Create the Beta-VAE model
def CreateModels(nl, b, inp,call1):
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
        lr = 1e-5
        if epoch > 35:
            print("New learning rate")
            lr = 1e-6
        if(epoch > 75):
            lr = 1e-8
        return lr

    scheduler = LearningRateScheduler(lr_scheduler)
    #Define adam optimizer
    adam = keras.optimizers.Adam(lr=1e-6, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    autoencoder.compile(optimizer='adam',loss=vae_loss, metrics=[vae_loss])
    hist = train_wo_s(inp,autoencoder,scheduler,2,16,call1)
    #SaveAutoencoderModel(autoencoder,dir_path)
    #SaveEncoderModel(encoder,dir_path)

    return hist, autoencoder, encoder, z_log_var

#Train the model without saving for hyperparameter tuning.
def train_wo_s(X,autoencoder,scheduler,epoch_number,batch_size_number,call1):
    X_train,X_test = X
    callbacks_list = [EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=0),call1,scheduler] #callback list
    hist=autoencoder.fit(X_train, X_train,epochs=epoch_number,batch_size=batch_size_number,shuffle=True,validation_data=(X_test, X_test), callbacks=callbacks_list, verbose=2)
    return hist

def store_iter_result(nl,b,value):
    csvfile = open('/home/scope/Carla/B-VAE-OOD-Monitor/hyperparameter-tuning/elbo/results' + '/iter-trial-result-50.csv' ,'a')
    writer = csv.writer(csvfile)
    val=[]
    val.append(nl)
    val.append(b)
    val.append(value)
    writer.writerow(val)

#Hyperparameter search
def hyperparameter_search(combinations_list,inp,train_data,call1):
        scores = []
        latent_inputs = []
        beta_inputs = []
        parameters=[]
        i=1
        for combination in combinations_list:

            nl = combination[0]
            b =  combination[1]
            b = round(b,2)

            print('-----------------------')
            print('Iteration #{}'.format(i))
            print("Number of latent units: {}".format(nl))
            print("beta value: {}".format(round(b,2)))

            loss_val, autoencoder, encoder, z_log_var = CreateModels(nl,b,inp,call1)
            latent_inputs.append(nl)
            beta_inputs.append(b)
            parameters.append([nl,b])
            scores.append(loss_val.history['loss'][-1])
            #scores.append(MIG)
            store_iter_result(nl,b,loss_val.history['loss'][-1])
            K.clear_session()
            i+=1

        #min_score_index = np.argmin(scores)
        min_score_index = np.argmin(scores)
        return parameters, parameters[min_score_index],scores,latent_inputs,beta_inputs


def plot(latent_inputs,beta_inputs,Latents,betas,scores):
    fig = plt.figure(figsize = (6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(latent_inputs, beta_inputs,
               scores, c = scores,
               cmap = plt.cm.seismic_r, s = 40)

    ax.set_xlabel('nl')
    ax.set_ylabel('B')
    ax.set_zlabel('ELBO')

    plt.title('Score as Function of nl and Beta')
    plt.savefig('/home/scope/Carla/B-VAE-OOD-Monitor/hyperparameter-tuning/elbo/results/random.png')
    plt.show()

def store_data(latent_inputs,beta_inputs,scores):
    csvfile = open('/home/scope/Carla/B-VAE-OOD-Monitor/hyperparameter-tuning/elbo/results' + '/random-elbo-result-50.csv' ,'a')
    writer = csv.writer(csvfile)
    for i in range(len(scores)):
        auto=[]
        auto.append(latent_inputs[i])
        auto.append(beta_inputs[i])
        auto.append(scores[i])
        writer.writerow(auto)


if __name__ == '__main__':
    train_data = "/home/scope/Carla/CARLA_0.9.6/PythonAPI/SVDD/data-generator/Trial4/"
    comp_inp,input_image = load_training_images(train_data)
    input_image = shuffle(input_image)
    inp = data_reshape(input_image)
    call1 = new_callback()
    call2 = MemoryCallback()
    Latents = np.arange(30, 200,10)
    betas = np.arange(1.0, 5.0, 0.1)
    n_iter=2
    combinations_list = [list(x) for x in product(Latents,betas)]
    random_combinations_index = np.random.choice(range(0,len(combinations_list)), n_iter, replace=False)
    combinations_random_chosen = [combinations_list[x] for x in random_combinations_index]
    #print(combinations_random_chosen)
    print("******************************Hyperparameter Tuning******************************************")
    t1= time.time()
    parameter, best_parameter,scores,latent_inputs,beta_inputs = hyperparameter_search(combinations_random_chosen,inp,train_data,call1)
    print(time.time()-t1)
    store_data(latent_inputs,beta_inputs,scores)
    t = PrettyTable(['Parameters','ELBO'])
    for i in range(len(parameter)):
        t.add_row([parameter[i],scores[i]])
    print(best_parameter)
