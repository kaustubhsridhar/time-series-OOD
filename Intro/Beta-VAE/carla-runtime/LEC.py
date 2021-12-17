#!/usr/bin/env python3
import random
from collections import deque
import numpy as np
import cv2
import time
import tensorflow as tf
import keras.backend.tensorflow_backend as backend
from keras.models import load_model
from util import CarEnv
import zmq
import base64
import carla
import sys
import os
import resource
import psutil
import csv
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
maxy=1
miny=-1
MODEL_PATH = '/home/scope/Carla/CARLA_0.9.6/PythonAPI/EMSOFT2020/model-weights/weights.best.hdf5'
port1 = '5005'
port2 = '5007'

def pubSocket(port2):
    context = zmq.Context()
    socket1= context.socket(zmq.PUB)
    socket1.bind("tcp://*:%s"%port2)
    return socket1

#Creating ZMQ Subscriber socket
def subSocket(port1):
    context = zmq.Context()
    socket2= context.socket(zmq.SUB)
    socket2.connect("tcp://localhost:%s" % port1)
    return socket2

def preprocess(current_state):
    img = cv2.resize(current_state, (200, 66))
    img = img/255.
    return img

def LEC(model,socket1,socket2,env):
    socket2.setsockopt(zmq.SUBSCRIBE, b'')
    while True:
        try:
            val=[]
            data = socket2.recv_string()
            time1=time.time()
            img = base64.b64decode(data)
            npimg = np.fromstring(img, dtype=np.uint8)
            frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            frame= preprocess(frame)
            inputs = np.array(frame)[np.newaxis]
            steer= model.predict(inputs, batch_size=1)
            steer=(float(steer[0][0])*(maxy-miny))+miny
            steer = round(steer,2)

            time2=time.time()-time1
            socket1.send_string(str(steer))
            cpu = psutil.cpu_percent()
            # gives an object with many fields
            mem = psutil.virtual_memory()
            #val.append(cpu)
            val.append(time2)

            # with open('LEC-time.csv', 'a') as file:
            #     writer = csv.writer(file)
            #     writer.writerow(val)
        except KeyboardInterrupt:
            socket1.close()
            socket2.close()
            sys.exit()
            print('done.')

if __name__ == '__main__':
    socket1 = pubSocket(port2)#Publisher sockets
    socket2 = subSocket(port1)#subscriber sockets
    env = CarEnv()
    model = load_model(MODEL_PATH)# Load the model
    LEC(model,socket1,socket2,env)
