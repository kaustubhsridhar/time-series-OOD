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
from glob import glob
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def load_model(model_path):
    print('model path is: ', model_path)
    with open(model_path + 'en_model.json', 'r') as jfile:
            model_svdd = model_from_json(jfile.read())
    model_svdd.load_weights(model_path + 'en_model.h5')
    return model_svdd

#Load complete input images without shuffling
def load_images(path):
    inputs = []
    locs = glob(path + "*")
    for idx, scenefolder in enumerate(locs):
        for img in sorted(glob(scenefolder + "/*.png")):
            img = cv2.imread(img)
            img = cv2.resize(img, (224, 224))
            img = img / 255.
            inputs.append(img)
    print('loaded test images.')
    return inputs


#Load complete input images without shuffling
def load_calib_images(root_dir):
    inputs = []
    train_folders = [12, 15, 14, 2, 16, 18, 0, 25, 9, 23, 28, 22, 11]
    folder_locs = []
    for folder_number in train_folders:
        if folder_number <= 10:
            folder_locs.append(root_dir+"setting_1/"+str(folder_number))
        elif folder_number >= 11 and folder_number <= 21:
            folder_locs.append(root_dir+"setting_2/"+str(folder_number-11))
        elif folder_number >= 22 and folder_number <= 32:
            folder_locs.append(root_dir+"setting_3/"+str(folder_number-22))

    for idx, scenefolder in enumerate(folder_locs):
        for img in sorted(glob(scenefolder + "/*.png")):
            img = cv2.imread(img)
            img = cv2.resize(img, (224, 224))
            img = img / 255.
            inputs.append(img)
    print('loaded calibration images.')
    return inputs


def test_runtime(autoencoder,img):
    #inputs = np.array([img])
    autoencoder_res = np.array(autoencoder.predict(img))
    autoencoder_res = Monitor_Helper.results_transpose(autoencoder_res)
    return autoencoder_res

def get_calibration_kl(model_path, calib_data_path, class_detector):
    X_calib = load_calib_images(calib_data_path)
    model_vae=load_model(model_path)
    kl_value = []
    for i in range(0,len(X_calib)):
        img = np.array(X_calib[i])[np.newaxis]
        test_val = []
        test_mean_data = test_runtime(model_vae, img)
        test_val.append(test_mean_data[0][0].tolist())
        test_val.append(test_mean_data[0][1].tolist())
        test_mean_dist=Monitor_Helper.test_data_extractor(test_val,latentsize)
        kl,kl_avg = Monitor_Helper.kl_computation(test_mean_dist,class_detector)
        # print(f'kl:     {kl}')
        # print(f'kl_avg: {kl_avg}')
        kl_value.append(kl_avg)
    print('loaded computed calibration kl"s.')
    return kl_value



def vae_prediction(model_path,data_path, iD_data_path,kl_value1,class_detector, folder_name): #,test_data_path,folder,model):
    print("==============PREDICTING THE LABELS ==============================")
    OOD_data = load_images(data_path) 
    iD_data = load_images(iD_data_path)
    X_validate = OOD_data + iD_data
    GTs = [0 for _ in range(len(OOD_data))] + [1 for _ in range(len(iD_data))]
    save_npz = True
    sliding_window = 6 #20 #Martingale sliding window
    scores_of_OOD_only = []
    scores_of_iD_only = []

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

        float_m_val = math.log(m[0])
        if GTs[i] == 0:
            scores_of_OOD_only.append(-1*float_m_val)
        elif GTs[i] == 1:
            scores_of_iD_only.append(-1*float_m_val)

    try:
        os.mkdir(f'../npz_saved_sliding_{sliding_window}/')
    except:
        pass
    np.save(f'../npz_saved_sliding_{sliding_window}/{folder_name}_win_out_Beta-VAE', scores_of_OOD_only)
    np.save(f'../npz_saved_sliding_{sliding_window}/{folder_name}_win_in_Beta-VAE', scores_of_iD_only)

    print("Total Anomalies:%d"%(total_anomalies)) #Total OOD detected in the scene
    # print("Total Detection Time:%f"%mean(time_val)) #Average detection time across the scene
    # print("Total CPU Utilization:%f"%mean(cpu_util)) #Average CPU utilization percentage across the scene
    # print("Total Mem Utilization:%f"%mean(mem_util)) #Average Memory utilization percentage across the scene




if __name__ == '__main__':
    models = ["30_1.4"] #,"30_1.1"
    test_data_path =  "../../carla_data/testing/" # "/home/scope/Carla/CARLA_0.9.6/PythonAPI/TCPS-data/Test-data/"
    test_folders = ["replay", "snowy", "foggy", "night", "rainy"] # ["in-distribution","heavy-rain","high-brightness","heavy-rain-70"]
    path = "../carla_models/" # "/home/scope/Carla/CARLA_0.9.6/PythonAPI/TCPS-results/Latent-extraction/"
    iD_test_data_path = "../../carla_data/testing/in/"
    calib_data_path = "../../carla_data/training/"
    for model in models:
        kl_value = []
        model_path = path + model + '/'
        print("----------------%s----------------"%model)
        if(model == "30_1.4"):
            latentsize = 30
            class_detector = [0,2,20,25,29]
        if(model == "30_1.2"):
            latentsize = 30
            class_detector =  [21,3,26,19,4]
        # train_data_path = path + model + '/' + "train-latents.csv"
        # calib_data_path = path + model + '/' + "caliberation-latents.csv"
        # csv_file = path + model + '/' + 'calib_kl.csv'
        # with open(csv_file, 'r') as file:
        #     reader = csv.reader(file)
        #     for row in reader:
        #         kl_value.append(row)
        kl_value = get_calibration_kl(model_path, calib_data_path, class_detector)

        kl_value1 = []
        for i in range(len(class_detector)):
            kl_value1.append(kl_value[class_detector[i]])
        for folder in test_folders:
            #anomaly_path = "/home/scope/Carla/CARLA_0.9.6/PythonAPI/TCPS-results/anomaly-results/" + model + '/'
            # anomaly_path = path + model + '/'
            # os.makedirs(anomaly_path, exist_ok=True)
            data_path = test_data_path + "out_" + folder + "/out/"
            vae_prediction(model_path,data_path, iD_test_data_path,kl_value1,class_detector, folder) # ,anomaly_path,folder,model) # Use M value for AUROC
