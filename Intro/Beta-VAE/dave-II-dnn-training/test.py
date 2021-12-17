#!/usr/bin/env python3
#libraries
import numpy as np
from keras.models import model_from_json
import csv
import glob
import cv2
import os
import time
seed = 7
np.random.seed(seed)

os.environ["CUDA_VISIBLE_DEVICES"]="1"#Setting the script to run on GPU:1,2

def nncontroller(img, model):
    inputs = np.array(img)[np.newaxis]
    outputs = model.predict(inputs, batch_size=1)
    return float(outputs[0][0])

def predict(data_path, validationFolders,model):
        miny=-1
        maxy=1
        csvfile = open(model_path + "steer_predictions.csv", "w")
        writer = csv.writer(csvfile)
        for folder in validationFolders:
            numFiles = len(glob.glob1(data_path + folder,'*.png'))
            print(numFiles)
            print("Total number of images collected: %d" %numFiles)
            for i in range(numFiles):
                x=[]
                data=[]
                image=cv2.imread(data_path + folder + '/frame%d.png'%i)
                img = cv2.resize(image, (200, 66))
                img = img / 255.
                time1= time.time()
                steer = nncontroller(img, model)
                time2=time.time()
                pred_time = time2 -time1
                print(pred_time)
                steering=(float(steer)*(maxy-miny))+miny
                steering=round(steering, 2)
                data.append(pred_time)
                data.append(steering)
                writer.writerow(data)

if __name__ == '__main__':
    data_path = "/home/scope/Carla/B-VAE-OOD-Monitor/data-generation/results/"
    model_path = "/home/scope/Carla/B-VAE-OOD-Monitor/LEC/results/"
    validationFolders = ["scene0"]
    with open(model_path + 'model.json', 'r') as jfile:
        model = model_from_json(jfile.read())
    model.load_weights(model_path + 'weights.best.hdf5')
    predict(data_path, validationFolders,model)
