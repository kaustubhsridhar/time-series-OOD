#!/usr/bin/env python2

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

def results_transpose(res):
    ret = []
    for i in range(res.shape[1]):
        ret.append(res[:, i, :].copy())
    return np.array(ret)


def test_data_extractor(test_mean_data,latentsize):
    distribution = []
    a=[]
    for i in range(2):
        distribution.append([])
        a.append([])
        #m.append([])

    for i in range (2):
        for j in range(latentsize):
            distribution[i].append([])

    for x in range(2):
        data = test_mean_data[x]
        #data[0] = data[0][1:]
        #data[len(data)-1]=data[len(data)-1][:-1]
        data = np.array(data)
        #print(data[0])
        for y in range (latentsize):
            distribution[x][y].append(float(data[y]))

    return distribution


def mse_computer(Working_path,Fadress,x,k,latentsize):
    distribution = []
    a = []
    m = []

    for i in range(2):
        distribution.append([])
        a.append([])
        #m.append([])

    for i in range (2):
        for j in range(latentsize):
            distribution[i].append([])

    for x in range(2):
        with open(Fadress, 'rt') as csvfile:
              reader = csv.reader(csvfile)
              for row in reader:
                  data = row[x].strip().split(',')
                  #data = row[x]
                  data[0] = data[0][1:]
                  data[len(data)-1]=data[len(data)-1][:-1]
                  data = np.array(data)
                  for y in range (latentsize):
                      distribution[x][y].append(float(data[y]))

    return distribution

def var_to_std(train_mean_dist):
    std = []
    for j in range(30):
        std.append([])

    for i in range(30):
        for k in range(len(train_mean_dist[0][0])):
            curLogVar = float(train_mean_dist[1][i][k])
            curvar = math.exp(curLogVar)
            curstd = math.sqrt(curvar)
            std[i].append(curvar)

    return(std)

def mse_pvalue_computer(dist1_mean,dist1_std,dist2_mean,latentsize):

    mse_avg=[]
    mse=[]
    total_avg=[]
    for z in range(latentsize):
        z_mse = []
        mse = 0.0
        for k in range(len(dist2_mean[0][0])):
            sample_mse = []
            for j in range(len(dist1_mean[0][0])-3000):
                mse = np.square(np.subtract(dist2_mean[0][z][k],dist1_mean[0][z][j])) #dist1_mean[0][z][j]
                sample_mse.append(mse)
            mse_val = min(sample_mse)
            z_mse.append(mse_val)
        mse_avg.append(z_mse)
        total_avg.append(sum(z_mse)/len(z_mse))

    return mse_avg,total_avg

def mse_pvalue_test(dist1_mean,dist1_std,dist2_mean,class_detector):

    mse_avg=[]
    mse=[]
    total_avg=[]
    for z in class_detector:
        z_mse = []
        mse = 0.0
        for k in range(len(dist2_mean[0][0])):
            sample_mse = []
            for j in range(len(dist1_mean[0][0])-3000):
                mse = np.square(np.subtract(dist2_mean[0][z][k],dist1_mean[0][z][j])) #dist1_mean[0][z][j]
                sample_mse.append(mse)
            mse_val = min(sample_mse)
            z_mse.append(mse_val)
        mse_avg.append(z_mse)
        total_avg.append(sum(z_mse)/len(z_mse))

    return mse_avg,total_avg


def integrand(k,p_anomaly):
    result = 1.0
    for i in range(len(p_anomaly)):
        result *= k*(p_anomaly[i]**(k-1.0))
    return result

def kl_divergence(Q, P):
     epsilon = 0.00001
     P = P+epsilon
     Q = Q+epsilon

     divergence = np.sum(P*np.log(P/Q))
     return divergence

def KL_computer(Fadress,x,k,class_detector):
    latentsize=30
    train_distribution = []
    a = []
    m = []

    for i in range(2):
        train_distribution.append([])
        a.append([])
        #m.append([])

    for i in range (2):
        for j in range(latentsize):
            train_distribution[i].append([])

    for x in range(2):
        with open(Fadress, 'rt') as csvfile:
              reader = csv.reader(csvfile)
              for row in reader:
                  data = row[x].strip().split(',')
                  data[0] = data[0][1:]
                  data[len(data)-1]=data[len(data)-1][:-1]
                  data = np.array(data)
                  for y in range (latentsize):
                      train_distribution[x][y].append(float(data[y]))
    kl_avg=[]
    kl=[]
    #print(len(train_distribution[0][0]))
    for z in class_detector:
        avg_klloss = []
        kl_val = 0.0
        for k in range(len(train_distribution[0][0])):
            mean = train_distribution[0][z][k]
            logvar = train_distribution[1][z][k]
            sd = math.sqrt(math.exp(logvar))
            x = np.arange(-10, 10, 0.001)
            p = norm.pdf(x, 0, 1)  # Normal Curve
            sum_p = np.sum(p)
            p[:] = [y / sum_p for y in p]
            q = norm.pdf(x, mean, sd) #
            sum_q = np.sum(q)
            q[:] = [z / sum_q for z in q]
            #q = norm.pdf(x, mean, sd) #
            #distance.mahalanobis(q,p)
            klloss = kl_divergence(q, p)
            avg_klloss.append(klloss)
        kl_val = sum(avg_klloss)/len(avg_klloss)
        #kl_value= float(avg_klloss/len(train_distribution[0][0]))
        #print(kl_val)
        kl.append(avg_klloss)
        kl_avg.append(kl_val)

    return kl,kl_avg


def kl_computation(train_distribution,class_detector):

    kl_avg=[]
    kl=[]
    #print(len(train_distribution[0][0]))
    for z in class_detector:
        avg_klloss = []
        kl_val = 0.0
        for k in range(len(train_distribution[0][0])):
            mean = train_distribution[0][z][k]
            logvar = train_distribution[1][z][k]
            sd = math.sqrt(math.exp(logvar))
            x = np.arange(-10, 10, 0.001)
            p = norm.pdf(x, 0, 1)  # Normal Curve
            sum_p = np.sum(p)
            p[:] = [y / sum_p for y in p]
            q = norm.pdf(x, mean, sd) #
            sum_q = np.sum(q)
            q[:] = [z / sum_q for z in q]
            #q = norm.pdf(x, mean, sd) #
            #distance.mahalanobis(q,p)
            klloss = kl_divergence(q, p)
            avg_klloss.append(klloss)
        kl_val = sum(avg_klloss)/len(avg_klloss)
        #kl_value= float(avg_klloss/len(train_distribution[0][0]))
        #print(kl_val)
        kl.append(avg_klloss)
        kl_avg.append(kl_val)

    return kl,kl_avg
