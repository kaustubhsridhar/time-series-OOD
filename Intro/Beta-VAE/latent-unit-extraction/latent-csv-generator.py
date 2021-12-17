#!/usr/bin/env python3
#input: script uses the trained B-VAE network to generate latent variable parameters (mean,logvar, and samples) which are stored in a csv.
#output: A csv file with row[0] -  mean, row[1] - logvar and row[2] - samples. and plots a scatter plot of the latent distributions.

#Libraries
import random
import os
import sys
import cv2
import csv
import glob
import numpy as np
import time
import numpy as np
from keras.models import model_from_json
from sklearn.metrics import mean_squared_error
from skimage import measure
import matplotlib
import matplotlib.pyplot as plt
from numpy.random import seed
from tensorflow import set_random_seed
set_random_seed(2)
os.environ["CUDA_VISIBLE_DEVICES"]="2"#Setting the script to run on GPU:1,2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def results_transpose(res):
    ret = []
    for i in range(res.shape[1]):
        ret.append(res[:, i, :].copy())
    return np.array(ret)

#Test script which uses either encoder or autoencoder.
#Full:Use entire dataset at once for prediction, Partial:Use step prediction iterating through induvidual images.
#Autoencoder:Reconstructs the images, Encoder:Generates the latent space data.
def test(autoencoder,img,Faddress):
        csvfile = open(Faddress,'a+')
        writer = csv.DictWriter(csvfile,fieldnames=['mean', 'logvar', 'sample'])
        autoencoder_res = np.array(autoencoder.predict(img))
        autoencoder_res = results_transpose(autoencoder_res)
        print(len(autoencoder_res))
        for i in range(len(autoencoder_res)):
            auto={}
            auto['mean'] = autoencoder_res[i][0].tolist()
            auto['logvar'] = autoencoder_res[i][1].tolist()
            auto['sample'] = autoencoder_res[i][2].tolist()
            writer.writerow(auto)

#Load complete input images without shuffling
def load_images(folder_path,Folders):
    numImages = 0
    inputs = []
    path = folder_path + Folders + '/'
    print(path)
    numFiles = len(glob.glob1(path,'*.png'))
    numImages += numFiles
    print("Total number of images:%d" %(numImages))
    for img in glob.glob(path+'*.png'):
        img = cv2.imread(img)
        img = cv2.resize(img, (224, 224))
        img = img / 255.
        inputs.append(img)
    return inputs

#Function which reads a dataset and clusters it.
#For this create a unified dataset with training and testing dataset.
#This also writes to which cluster the latent data belons to.
def plotting(Working_path,dataset,latentsize,folder):
    train_distribution = []
    a = []
    m = []

    for i in range(2):
        train_distribution.append([])
        a.append([])

    for i in range (2):
        for j in range(latentsize):
            train_distribution[i].append([])

    for x in range(2):
        with open(dataset, 'rt') as csvfile:
              reader = csv.reader(csvfile)
              for row in reader:
                  data = row[x].strip().split(',')
                  data[0] = data[0][1:]
                  data[len(data)-1]=data[len(data)-1][:-1]
                  data = np.array(data)
                  for y in range (latentsize):
                      train_distribution[x][y].append(float(data[y]))
    k=[]
    for z in range(len(train_distribution[0])):
            a[0].append(train_distribution[0][z])
            a[1].append(train_distribution[1][z])

            m=[]
            for x in range(len(train_distribution[0][0])):
                m.append(z)
            k.append(m)

    plt.scatter(a[0],a[1], c=k, s=20, cmap='viridis')
    plt.xlabel('Mean')
    plt.ylabel('LogVar')
    #plt.title('Z plots for ')
    #plt.range(-0.5,0.5)
    plt.colorbar()
    #path = Working_path + '/calibration-test'
    plt.savefig(Working_path + folder + '.png', bbox_inches='tight')
    plt.clf()
    plt.close()
    #plt.show()


if __name__ == '__main__':
    models = ["30_1.4"] #["30_1.0","30_1.2",
    run = "brightness"
    #Load image dataset to be plotted
    path =  "/home/scope/Carla/CARLA_0.9.6/PythonAPI/TCPS-data/Test-data/"   #latent-data/"
    Folders = ["anomaly-detection"] #["s0","p25","p50","b25","b50","p50b50"]
    for model in models:
        if(model == "30_1.0" or model == "30_1.2" or model == "30_1.4" or model == "30_1.1"):
            x = 30
        if(model == "40_1.0" or model == "40_1.5"):
            x = 40
        Working_folder = "/home/scope/Carla/CARLA_0.9.6/PythonAPI/TCPS-results/Latent-extraction/" + model + '/'
        model_weights = "/home/scope/Carla/CARLA_0.9.6/PythonAPI/TCPS-results/Trained-models-new/"  + model + "/"
        os.makedirs(Working_folder, exist_ok=True)
        for folder in Folders:
            image = load_images(path,folder)
            img = np.array(image[0:len(image)].copy())#test data
            img = np.reshape(img,[-1, 224,224,3])#Reshape the data
            File = folder + '.csv'
            Fadress = Working_folder + File
            with open(model_weights + 'en_model.json', 'r') as jfile:
                autoencoder = model_from_json(jfile.read())
            autoencoder.load_weights(model_weights + 'en_model.h5')
            test(autoencoder,img,Fadress)#Run the test script
            plotting(Working_folder,Fadress,x,folder)
