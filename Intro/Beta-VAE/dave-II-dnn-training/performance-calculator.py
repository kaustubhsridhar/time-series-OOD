#!/usr/bin/env python3
#libraries
import csv
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

def mse_calculator(actual,predicted):
    data1=[]
    data2=[]
    x=[]
    time=[]
    i=0
    with open(actual,'r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            i += 1
            time.append(float(i*0.1))
            data1.append(float(row[0]))#predicted
            x.append(float(i))

    with open(predicted,'r') as csvfile:
        plots = csv.reader(csvfile)
        for row in plots:
            i += 1
            data2.append(float(row[1]))#predicted
            x.append(float(i))

    print(len(data1))
    print(len(data2))

    mse = mean_squared_error(data1,data2)
    print(mse)

    return data1, data2, time

def plotter(data1,data2,time,path):
    plt.figure(figsize=(8,3))
    plt.plot(time,data1,color='brown')
    plt.plot(time,data2,color='green')
    #plt.plot(time,data3,linestyle='--',color='orange')
    plt.xlabel('Time (s)',fontsize=14)
    plt.ylabel('Normalized \nSteering Angle',fontsize=14)
    #plt.title('predicted vs actual steering')
    plt.legend(['AutoPilot Steering', 'LEC Steering'], loc='upper center', bbox_to_anchor=(0.5, -0.2),
              ncol=3, fancybox=True, fontsize=12, shadow=True)
    plt.savefig(path + 'plot.png', bbox_inches='tight')
    #plt.show()

if __name__ == '__main__':
    actual_steer = "/home/scope/Carla/B-VAE-OOD-Monitor/data-generation/results/scene0/steer.csv"
    predicted_steer = "/home/scope/Carla/B-VAE-OOD-Monitor/LEC/results/steer_predictions.csv"
    save_path = "/home/scope/Carla/B-VAE-OOD-Monitor/LEC/results/"
    data1,data2,time = mse_calculator(actual_steer,predicted_steer)
    plotter(data1,data2,time,save_path)
