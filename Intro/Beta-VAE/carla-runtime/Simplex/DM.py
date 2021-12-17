#!/usr/bin/env python2

import random
from collections import deque
import numpy as np
import cv2
import time
from util import CarEnv
import zmq
import base64
import carla
import sys
import psutil
import csv
maxy=1
miny=-1
port2 = '5006'
port3 = '5007'
port4 = '5008'
port5 = '5010'
port6 = '5011'

def pubSocket(port4):
    context = zmq.Context()
    socket1= context.socket(zmq.PUB)
    socket1.bind("tcp://*:%s"%port4)
    return socket1

#Creating ZMQ Subscriber socket
def subSocket(port2,port3,port5,port6):
    context = zmq.Context()
    socket2= context.socket(zmq.SUB)
    socket2.connect("tcp://localhost:%s" % port2)
    socket3= context.socket(zmq.SUB)
    socket3.connect("tcp://localhost:%s" % port3)
    socket4 = context.socket(zmq.SUB)
    socket4.connect("tcp://localhost:%s" % port5)
    socket5 = context.socket(zmq.SUB)
    socket5.connect("tcp://localhost:%s" % port6)
    return socket2,socket3, socket4,socket5

def DM(socket1,socket2,socket3,socket4,socket5):
    socket2.setsockopt(zmq.SUBSCRIBE, b'')
    socket3.setsockopt(zmq.SUBSCRIBE, b'')
    socket4.setsockopt(zmq.SUBSCRIBE, b'')
    #socket5.setsockopt(zmq.SUBSCRIBE, b'')
    LEC_prev = 0.0
    while True:
        try:
            val=[]
            safety_steer = socket2.recv_string()
            print(safety_steer)
            LEC_steer = socket3.recv_string()
            print(LEC_steer)
            #feature = socket5.recv_string()
            #feature[0],feature[1] = feature.split()
            confidence = socket4.recv_string()
            print("confidence:%s"%confidence)
            time1=time.time()
            #print("feature1:%s, feature2:%s"%(feature[0], feature[1]))

            if(abs(float(LEC_steer) - LEC_prev)>0.05):
                LEC_steer = LEC_prev
            else:
                LEC_steer = LEC_steer

            if(abs(float(LEC_steer) - float(safety_steer))>0.05):
                steer = safety_steer
            else:
                steer = LEC_steer
            #print(steer)
            time2=time.time()-time1
            socket1.send_string(str(steer))
            print("sent data")
            cpu = psutil.cpu_percent()
            # gives an object with many fields
            mem = psutil.virtual_memory()
            #val.append(cpu)
            val.append(time2)
            with open('DM-time.csv', 'a') as file:
                writer = csv.writer(file)
                writer.writerow(val)
        except KeyboardInterrupt:
            socket1.close()
            socket2.close()
            socket3.close()
            socket4.close()
            sys.exit()
            print('done.')

if __name__ == '__main__':
    socket1 = pubSocket(port4)#Publisher sockets
    socket2,socket3,socket4,socket5 = subSocket(port2,port3,port5,port6)#subscriber sockets
    DM(socket1,socket2,socket3,socket4,socket5)
