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
import MIG_utils
from statistics import mean,median
from keras.callbacks import TerminateOnNaN

plt.style.use('ggplot')
warnings.filterwarnings("ignore")

import resource

class MemoryCallback(Callback):
    def on_epoch_end(self, epoch, log={}):
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


os.environ["CUDA_VISIBLE_DEVICES"]="2"#Setting the script to run on GPU:1,2
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
    hist = train_wo_s(inp,autoencoder,scheduler,50,16,call1)
    #SaveAutoencoderModel(autoencoder,dir_path)
    #SaveEncoderModel(encoder,dir_path)

    return hist, autoencoder, encoder, z_log_var

#Train the model without saving for hyperparameter tuning.
def train_wo_s(X,autoencoder,scheduler,epoch_number,batch_size_number,call1):
    X_train,X_test = X
    callbacks_list = [EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=0),call1,scheduler] #callback list
    hist=autoencoder.fit(X_train, X_train,epochs=epoch_number,batch_size=batch_size_number,shuffle=True,validation_data=(X_test, X_test), callbacks=callbacks_list, verbose=2)
    return hist

def takeFirst(elem):
    return elem[1]

def takeSecond(elem):
    return elem[0]

def MIG_calculation(csv_path,latentspacesize,size,numgenerative,sampling_value,iterations):
    Datasetsize = size # number of images in the dataset
    MIG_Values = []
    latent_Mutual_info = []
    csv_value=[]
    B_Latent_info = []
    MIG_in_iteration = []
    Final_Mutual_info = []
    Latent_Ranking_in_iteration = []
    for j in range(numgenerative):
            latent_Mutual_info.append([])
    for j in range(numgenerative):
            Latent_Ranking_in_iteration.append([])
    for j in range(numgenerative):
            Final_Mutual_info.append([])

    for j in range(numgenerative):
        for k in range(latentspacesize):
            Final_Mutual_info[j].append([])

    print("**************Computing MIG*************")
    #iteration loop of an Experiment
    for g in range (iterations):
        print("**************Iteration:%d*************"%g)
        #extract latent distributions  (mean, variance, feature_variants)
        Mu,Logvar,feature_variants,samples = MIG_utils.MIG_compute(csv_path, Datasetsize, latentspacesize, numgenerative, sampling_value)

        #compute
        MIG,MIG_Tuple,Latent_Ranking = MIG_utils.Calculate_Entropy(sampling_value,Datasetsize, latentspacesize, Mu, Logvar,samples, feature_variants,numgenerative)
        print("Mutual Information Gap is %f"%(MIG))
        MIG_in_iteration.append(MIG)

        #Extract the ranking order from the MIG Tuple
        for j in range(numgenerative):
            Latent_Ranking_in_iteration[j].append(Latent_Ranking[j])

    for j in range(numgenerative):
        for k in range(latentspacesize):
            info = sum(Final_Mutual_info[j][k])/iterations
            value = (k,info)
            latent_Mutual_info[j].append(value)
    latent_Mutual_info.sort(key=takeFirst, reverse=False)
    MIG_in_iteration.sort(reverse=True)
    Median_Mig=median(MIG_in_iteration)

    #compute average MIG score across iterations in an experiment
    AVG_MIG = sum(MIG_in_iteration)/iterations
    print("Avg MIG is %f"%(AVG_MIG))

    return AVG_MIG

def results_transpose(res):
    ret = []
    for i in range(res.shape[1]):
        ret.append(res[:, i, :].copy())
    return np.array(ret)

def latent_generator(encoder,train_data,dir_path,nl,b,inp):
        inputs=[]
        label1=[]
        label2=[]
        csvfile = open(dir_path + '/nl%d_b%f_train_latent.csv'%(nl,b) ,'a')
        writer = csv.writer(csvfile)
        with open(train_data + 'train.csv', 'rt') as csvfile:
                reader = csv.reader(csvfile)
                for row in reader:
                    img = cv2.imread(train_data + row[0])
                    img = cv2.resize(img, (224, 224))
                    img = img / 255.
                    inputs.append(img)
                    label1.append(row[1])
                    label2.append(row[2])
        input_image = np.array(inputs)
        input_image = np.reshape(input_image, [-1, input_image.shape[1],input_image.shape[2],input_image.shape[3]])
        autoencoder_res = np.array(encoder.predict(input_image))
        autoencoder_res = results_transpose(autoencoder_res)
        for i in range(0,len(autoencoder_res),3):
            auto=[]
            auto.append(autoencoder_res[i][0].tolist())
            auto.append(autoencoder_res[i][1].tolist())
            #auto['sample'] = autoencoder_res[i][2].tolist()
            auto.append(label1[i])
            auto.append(label2[i])
            writer.writerow(auto)
        csv_path = dir_path + '/nl%d_b%f_train_latent.csv'%(nl,b)
        print(int(len(inputs)/3))
        return csv_path, int(len(inputs)/3)



def store_iter_result(nl,b,value):
    csvfile = open('/home/scope/Carla/B-VAE-OOD-Monitor/hyperparameter-tuning/mig/results' + '/iter-trial-result-301.csv' ,'a')
    writer = csv.writer(csvfile)
    val=[]
    val.append(nl)
    val.append(b)
    val.append(value)
    writer.writerow(val)

#Hyperparameter search
def hyperparameter_search(combinations_list,inp,dir_path,train_data,call1,mig_numgenerative,mig_sampling_value,mig_iterations):
        scores = []
        latent_inputs = []
        beta_inputs = []
        parameters=[]
        i=1
        #min_latents, max_latents = Latents
        #min_beta, max_beta = betas

        for combination in combinations_list:

            nl = combination[0]
            b =  combination[1]

            print('-----------------------')
            print('Iteration #{}'.format(i))
            print("Number of latent units: {}".format(nl))
            print("beta value: {}".format(round(b,2)))

            loss_val, autoencoder, encoder, z_log_var = CreateModels(nl,b,inp,call1)
            csv_path,size = latent_generator(encoder,train_data,dir_path,nl,b,inp)
            MIG = MIG_calculation(csv_path,nl,size,mig_numgenerative,mig_sampling_value,mig_iterations)
            os.remove(csv_path)
            latent_inputs.append(nl)
            beta_inputs.append(b)
            parameters.append([nl,b])
            scores.append(MIG)
            #scores.append(MIG)
            store_iter_result(nl,b,MIG)
            K.clear_session()
            i+=1

        #min_score_index = np.argmin(scores)
        max_score_index = np.argmax(scores)
        return parameters, parameters[max_score_index],scores,latent_inputs,beta_inputs


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
    plt.savefig('/home/scope/Carla/B-VAE-OOD-Monitor/hyperparameter-tuning/mig/results/random.png')
    plt.show()

def store_data(latent_inputs,beta_inputs,scores):
    csvfile = open('/home/scope/Carla/B-VAE-OOD-Monitor/hyperparameter-tuning/mig/results' + '/random-mig-result-301.csv' ,'a')
    writer = csv.writer(csvfile)
    for i in range(len(scores)):
        auto=[]
        auto.append(latent_inputs[i])
        auto.append(beta_inputs[i])
        auto.append(scores[i])
        writer.writerow(auto)


if __name__ == '__main__':
    train_data = "/home/scope/Carla/CARLA_0.9.6/PythonAPI/SVDD/data-generator/Trial4/"
    dir_path = '/home/scope/Carla/B-VAE-OOD-Monitor/hyperparameter-tuning/mig/results'
    comp_inp,input_image = load_training_images(train_data)
    input_image = shuffle(input_image)
    inp = data_reshape(input_image)
    call1 = new_callback()
    call2 = MemoryCallback()
    Latents = [30,200]#hyperparameter1
    betas = [1.1,5.0]#hyperparameter2
    mig_numgenerative = 2 # number of features in the image dataset
    mig_sampling_value = 100 #Number of random samples to compute empirical mutual information
    mig_iterations = 3 #number of iterations in an Experiment to compute average MIG score.
    min_latents, max_latents = Latents
    min_beta, max_beta = betas
    Latents = np.arange(min_latents, max_latents, 10)
    betas = np.arange(min_beta, max_beta, 0.1)
    n_iter=30
    combinations_list = [list(x) for x in product(Latents,betas)]
    random_combinations_index = np.random.choice(range(0,len(combinations_list)), n_iter, replace=False)
    combinations_random_chosen = [combinations_list[x] for x in random_combinations_index]
    print("******************************Hyperparameter Tuning******************************************")
    t1= time.time()
    parameter, best_parameter,scores,latent_inputs,beta_inputs = hyperparameter_search(combinations_random_chosen,inp,dir_path,train_data,call1,mig_numgenerative,mig_sampling_value,mig_iterations)
    print(time.time()-t1)
    store_data(latent_inputs,beta_inputs,scores)
    # t = PrettyTable(['Parameters','ELBO'])
    # for i in range(len(parameter)):
    #     t.add_row([parameter[i],scores[i]])
    # print(t)
    #print(parameter)
    print(best_parameter)
    plot(latent_inputs,beta_inputs,Latents,betas,scores)
    #print(scores)
