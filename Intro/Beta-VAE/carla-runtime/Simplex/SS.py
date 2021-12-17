#!/usr/bin/env python2
import random
from collections import deque
import numpy as np
from util import CarEnv
import cv2
import time
import zmq
import base64
import carla
import sys
import resource
import psutil
import csv

maxy=1
miny=-1
port1 = '5005'
port2 = '5006'

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

def LEC(socket1,socket2):
    # Reset environment and get initial state
    #current_state = env.reset()
    socket2.setsockopt(zmq.SUBSCRIBE, b'')
    while True:
        try:
            val=[]
            data = socket2.recv_string()
            time1=time.time()
            img = base64.b64decode(data)
            npimg = np.fromstring(img, dtype=np.uint8)
            frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
            time2=time.time() - time1
            safety_steer = 0.0
            socket1.send_string(str(safety_steer))
            cpu = psutil.cpu_percent()
            # gives an object with many fields
            mem = psutil.virtual_memory()
            #val.append(cpu)
            val.append(time2)

            # with open('SS-time.csv', 'a') as file:
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
    LEC(socket1,socket2)
