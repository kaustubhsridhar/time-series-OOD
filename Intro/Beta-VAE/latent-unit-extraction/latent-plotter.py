#!/usr/bin/env python3
#input: Trained B-VAE network and the dataset
#output: Scatter plots of induvidual latent variable distributions 
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
import matplotlib as mpl
import matplotlib.pyplot as plt
from numpy.random import seed
from tensorflow import set_random_seed
set_random_seed(2)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
os.environ["CUDA_VISIBLE_DEVICES"]="2"#Setting the script to run on GPU:1,2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
mpl.rcParams['axes.linewidth'] = 2 #set the value globally

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
    c = []

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
                      train_distribution[x][y].append(float(data[y])*10)

    for z in class_detector:
            a =[]
            k = []
            a.append(train_distribution[0][z])
            a.append(train_distribution[1][z])

            m=[]
            for x in range(len(train_distribution[0][0])):
                m.append(z)
            k.append(m)

            b = []
            for i in range(len(a[0])):
                c= []
                c.append(float(a[0][i]))
                c.append(float(a[1][i]))
                b.append(c)

            a = np.array(b)

            kmeans = KMeans(n_clusters=2, max_iter=1000, algorithm = 'auto',init='k-means++', n_init=20,random_state=0)#K means clustering
            fitted = kmeans.fit(a)#fit kmeans to PCA output
            alldistances = kmeans.fit_transform(a)#getting distance of the prediction from cluster centers
            #print(alldistances)
            totalDistance = np.min(alldistances, axis=1)
            totalDistance = np.array(totalDistance).reshape(-1,1)
            #print(totalDistance)#distance from the cluster center.
            prediction = kmeans.predict(a)#Predict the cluster number for each prediction
            # prediction = gmm.predict(a)#Predict the cluster number for each prediction
            #print(prediction)
            plt.figure(figsize=(6, 6),linewidth=5)#define a figure to plot
            centers = kmeans.cluster_centers_#get centers of each cluster
            #print(centers)
            print(silhouette_score(a, kmeans.labels_))
            c=0
            d=0

            #Plot the clusters
            LABEL_COLOR_MAP = {0 : 'y',1 : 'g'}
            label_color = [LABEL_COLOR_MAP[l] for l in kmeans.labels_]
            plt.scatter(a[:,0],a[:,1], s=50, cmap='viridis', c=label_color)
            plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5)
            #plt.axis('off')
            plt.xticks(fontsize=14)
            plt.yticks(fontsize=14)
            #plt.ylim([-2, 2])
            #plt.xlim([-2, 2])
            #plt.xlabel('Mean')
            #plt.ylabel('LogVar')
            #plt.colorbar()
            plt.savefig(Working_path +'plot%d.png'%z, bbox_inches='tight')

if __name__ == '__main__':
    models = ["30_1.4"] #["30_1.0","30_1.2",
    run = "brightness"
    #Load image dataset to be plotted
    path =  "/home/scope/Carla/CARLA_0.9.6/PythonAPI/TCPS-data/latent-data/"
    Folders = ["anomaly-detection"] #["s0","p25","p50","b25","b50","p50b50"]
    for model in models:
        if(model == "30_1.0" or model == "30_1.2" or model == "30_1.4" or model == "30_1.1"):
            x = 30
            class_detector = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]#[21,3,19,26]
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
            plotting(Working_folder,Fadress,x,folder)
