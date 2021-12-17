#!/usr/bin/env python3

import random
from collections import deque
import numpy as np
import cv2
import time
import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from keras.models import load_model
from keras.models import model_from_json
import zmq
import base64
import os
import matplotlib.pyplot as plt
import csv
from sklearn import preprocessing
import numpy as np
from scipy import stats
import math
#from math import log2
from scipy.stats import norm
import scipy.integrate as integrate
#from scipy.spatial import distance
import Monitor_Helper
import sys
from threading import Thread
import threading
import queue
import os
import plotTools
import psutil
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

maxy=1
miny=-1
MODEL_PATH = '/home/scope/Carla/CARLA_0.9.6/PythonAPI/EMSOFT2020/monitor-weights/'
port1 = '5005'
port2 = '5006'
port3 = '5007'
port4 = '5025'
port5 = '5026'
port6= '5021'

def pubSocket(port4,port5,port6):
    context = zmq.Context()
    socket1= context.socket(zmq.PUB)
    socket1.bind("tcp://*:%s"%port4)
    socket3= context.socket(zmq.PUB)
    socket3.bind("tcp://*:%s"%port5)
    socket4= context.socket(zmq.PUB)
    socket4.bind("tcp://*:%s"%port6)
    return socket1,socket3,socket4

#Creating ZMQ Subscriber socket
def subSocket(port1):
    context = zmq.Context()
    socket2= context.socket(zmq.SUB)
    socket2.connect("tcp://localhost:%s" % port1)
    return socket2

def preprocess(current_state):
    img = cv2.resize(current_state, (224, 224))
    img = img/255.
    return img

def test_runtime(autoencoder,img):
    inputs = np.array([img])
    #autoencoder_res = autoencoder.predict(inputs)
    autoencoder_res = np.array(autoencoder.predict(inputs))
    autoencoder_res = Monitor_Helper.results_transpose(autoencoder_res)
    #print(autoencoder_res)
    return autoencoder_res


def Monitor(autoencoder,socket1,socket2,socket3,socket4):
    class_detector = [25] #1
    #class2_detector = [20]
    #class_detectors = [class1_detector,class2_detector]
    sliding_window = 20
    latentsize = 30
    M=[]
    delta = 5
    threshold = 10
    state_val = []
    avg_pval = []
    anomaly_val=[]
    x=30
    k=0
    i=0
    time_val=[]
    p_anomaly=[]
    state_val1 = []
    avg_pval1 = []
    anomaly_val1=[]
    time_val1=[]
    p_anomaly1=[]
    start_time = time.time()
    Working_directory = '/home/scope/Carla/CARLA_0.9.6/PythonAPI/new/'
    Working_path = Working_directory + 'dataset' + '/'
    Fadress = Working_path + 'calibration.csv'
    print("processing kl value computation")
    kl_value1,kl_value1_avg = Monitor_Helper.KL_computer(Fadress,x,k,class_detector)
    print(time.time()-start_time)
    while True:
        try:
            val=[]
            #print("ready to receive image")
            time1 = time.time()
            socket2.setsockopt(zmq.SUBSCRIBE, b'')
            data = socket2.recv_string()
            img = base64.b64decode(data)
            npimg = np.fromstring(img, dtype=np.uint8)
            frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            frame = preprocess(frame)
            #print("received image")
            test_val = []
            test_mean_data = test_runtime(autoencoder, frame)
            test_val.append(test_mean_data[0][0].tolist())
            test_val.append(test_mean_data[0][1].tolist())
            test_mean_dist=Monitor_Helper.test_data_extractor(test_val,latentsize)
            avg_pval = 0.0
            pval=[]
            kl,kl_avg = Monitor_Helper.kl_computation(test_mean_dist,class_detector)
            for x in range(len(class_detector)):
                anomaly=0
                for l in range(len(kl_value1[0])):
                    if(kl[x] <= kl_value1[x-1][l]):
                        anomaly+=1
                p_value = anomaly/len(kl_value1[0])
            pval.append(p_value)
            final_pval = sum(pval)/len(pval)
            anomaly_val.append(final_pval)
            time_val.append(float(i*0.1))
            if(final_pval<0.005):
                p_anomaly.append(0.005)
            else:
                p_anomaly.append(final_pval)
            if(len(p_anomaly))>= sliding_window:
                p_anomaly = p_anomaly[-1*sliding_window:]
            m = integrate.quad(Monitor_Helper.integrand,0.0,1.0,args=(p_anomaly))
            m_val = int(math.log(m[0]))
            print(m_val)
            M.append(math.log(m[0]))
            final_pval = round(final_pval,2)
            if(i==0):
                S = 0
                S_prev = 0
            else:
                S = max(0, S_prev+m_prev-delta)
            S_prev = S
            m_prev = m[0]
            state_val.append(S)
            if(S > threshold):
                val1=1
            else:
                val1=0
            print("Anomaly:%d"%val1)
            #socket1.send_string(str(val))
            t = i*0.05
            socket4.send_string("%s %s %s %s"%(i,final_pval,m_val,val1))
            cpu = psutil.cpu_percent()
            # gives an object with many fields
            mem = psutil.virtual_memory()
            val.append(i)
            val.append(final_pval)
            val.append(m_val)
            val.append(val1)

            #res_limits = resource.getrusage(resource.RUSAGE_SELF)
            #print(res_limits)
            #print('Resouce Limits: ' + str(resource.getrlimit(resource.RLIMIT_CPU)))
            with open('EMSOFT/video/diagnoser-mixed.csv', 'a') as file:
                writer = csv.writer(file)
                writer.writerow(val)
            i+=1
            frame_time = time.time() - time1

        except KeyboardInterrupt:
            print('Cleaning up and terminating the script')
            time.sleep(3)
            sys.exit()
    #plotThread.join()


if __name__ == '__main__':
    socket1,socket3,socket4 = pubSocket(port4,port5,port6)#Publisher sockets
    socket2 = subSocket(port1)#subscriber sockets
    print('model loaded')
    with open(MODEL_PATH + 'en_model.json', 'r') as jfile:
        autoencoder = model_from_json(jfile.read())
    autoencoder.load_weights(MODEL_PATH + 'en_model.h5')
    Monitor(autoencoder,socket1,socket2,socket3,socket4)
