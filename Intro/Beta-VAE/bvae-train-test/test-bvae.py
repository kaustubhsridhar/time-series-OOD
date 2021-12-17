#!/usr/bin/env python
# coding: utf-8
#input: validation and test images
#output: Returns if the Image is an OOD.
import random
import os
import sys
import cv2
import csv
import glob
import numpy as np
import time
import psutil
import math
import base64
from sklearn.utils import shuffle
from keras.models import Model, model_from_json
import numpy as np
import tensorflow as tf
import os
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import scipy.integrate as integrate
import Monitor_Helper
from statistics import mean
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def load_model(model_path):
    with open(model_path + 'en_model.json', 'r') as jfile:
            model_svdd = model_from_json(jfile.read())
    model_svdd.load_weights(model_path + 'en_model.h5')
    return model_svdd

#Load complete input images without shuffling
def load_images(path):
    numImages = 0
    inputs = []
    numFiles = len(glob.glob1(path,'*.png'))
    numImages += numFiles
    for img in glob.glob(path+'*.png'):
        img = cv2.imread(img)
        img = cv2.resize(img, (224, 224))
        img = img / 255.
        inputs.append(img)
    return inputs


#Load complete input images without shuffling
def load_calibration_images(train_data):
    inputs = []
    comp_inp = []
    with open(train_data + 'calibration.csv', 'rt') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                img = cv2.imread(train_data + row[0])
                img = cv2.resize(img, (224, 224))
                img = img / 255.
                inputs.append(img)
            return inputs

def test_runtime(autoencoder,img):
    #inputs = np.array([img])
    autoencoder_res = np.array(autoencoder.predict(img))
    autoencoder_res = Monitor_Helper.results_transpose(autoencoder_res)
    return autoencoder_res

def vae_prediction(model_vae,data_path,kl_value1,class_detector,test_data_path,folder,model):
    print("==============PREDICTING THE LABELS ==============================")
    X_validate = load_images(data_path)
    sliding_window = 20 #Martingale sliding window
    M=[]
    delta = 5
    threshold = 20 #CUSUM threshold
    state_val = []
    avg_pval = []
    anomaly_val=[]
    k=0
    i=0
    time_val=[]
    p_anomaly=[]
    state_val1 = []
    avg_pval1 = []
    anomaly_val1=[]
    time_val1=[]
    p_anomaly1=[]
    tval = []
    dist_val=[]
    total_anomalies = 0
    anomaly_val = []
    time_val = []
    cpu_util = []
    mem_util = []
    model_vae=load_model(model_path)
    for i in range(0,len(X_validate)): #ICP computations
        val=[]
        img = np.array(X_validate[i])[np.newaxis]
        test_val = []
        test_mean_data = test_runtime(model_vae, img)
        t1 = time.time()
        test_val.append(test_mean_data[0][0].tolist())
        test_val.append(test_mean_data[0][1].tolist())
        test_mean_dist=Monitor_Helper.test_data_extractor(test_val,latentsize)
        avg_pval = 0.0
        pval=[]
        kl,kl_avg = Monitor_Helper.kl_computation(test_mean_dist,class_detector)
        frame_time = time.time() - t1
        for x in range(len(class_detector)):
            anomaly=0
            for l in range(len(kl_value1[0])):
                if(float(kl[x][0]) <= float(kl_value1[x][l])):
                    anomaly+=1
            p_value = anomaly/len(kl_value1[0])
        pval.append(p_value)
        final_pval = sum(pval)/len(pval)
        anomaly_val.append(final_pval)
        if(final_pval<0.005):
            p_anomaly.append(0.005)
        else:
            p_anomaly.append(final_pval)
        if(len(p_anomaly))>= sliding_window:
            p_anomaly = p_anomaly[-1*sliding_window:]
        m = integrate.quad(Monitor_Helper.integrand,0.0,1.0,args=(p_anomaly)) #Martingale computations
        m_val = int(math.log(m[0]))
        M.append(math.log(m[0])) #Log of martingale
        final_pval = round(final_pval,2)
        if(i==0):
            S = 0
            S_prev = 0
        else:
            S = max(0, S_prev+m_prev-delta) #CUSUM calculations
        S_prev = S
        m_prev = m[0]
        state_val.append(S)
        if(S > threshold):
            val1=1
            total_anomalies+=1
        else:
            val1=0

        # frame_time = time.time() - t1
        time_val.append(frame_time)
        cpu = psutil.cpu_percent()#cpu utilization stats
        mem = psutil.virtual_memory()#virtual memory stats
        cpu_mem = 100 * (mem.used/mem.total)
        cpu_util.append(cpu)
        mem_util.append(cpu_mem)

    print("Total Anomalies:%d"%(total_anomalies)) #Total OOD detected in the scene
    print("Total Detection Time:%f"%mean(time_val)) #Average detection time across the scene
    print("Total CPU Utilization:%f"%mean(cpu_util)) #Average CPU utilization percentage across the scene
    print("Total Mem Utilization:%f"%mean(mem_util)) #Average Memory utilization percentage across the scene


if __name__ == '__main__':
    models = ["30_1.4"] #,"30_1.1"
    test_data_path =  "/home/scope/Carla/CARLA_0.9.6/PythonAPI/TCPS-data/Test-data/"
    test_folders = ["in-distribution","heavy-rain","high-brightness","heavy-rain-70"]
    path = "/home/scope/Carla/CARLA_0.9.6/PythonAPI/TCPS-results/Latent-extraction/"
    for model in models:
        kl_value = []
        model_path = "/home/scope/Carla/CARLA_0.9.6/PythonAPI/TCPS-results/Trained-models-new/" + model + '/'
        print("----------------%s----------------"%model)
        if(model == "30_1.4"):
            latentsize = 30
            class_detector = [0,2,20,25,29]
        if(model == "30_1.2"):
            latentsize = 30
            class_detector =  [21,3,26,19,4]
        train_data_path = path + model + '/' + "train-latents.csv"
        calib_data_path = path + model + '/' + "caliberation-latents.csv"
        csv_file = path + model + '/' + 'calib_kl.csv'
        with open(csv_file, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                kl_value.append(row)
        kl_value1 = []
        for i in range(len(class_detector)):
            kl_value1.append(kl_value[class_detector[i]])
        for folder in test_folders:
            anomaly_path = "/home/scope/Carla/CARLA_0.9.6/PythonAPI/TCPS-results/anomaly-results/" + model + '/'
            os.makedirs(anomaly_path, exist_ok=True)
            data_path = test_data_path + folder + '/'
            vae_prediction(model_path,data_path,kl_value1,class_detector,anomaly_path,folder,model)
