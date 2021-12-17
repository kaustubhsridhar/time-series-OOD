#!/usr/bin/env python2
import random
from collections import deque
import numpy as np
import cv2
import time
from util import CarEnv
import zmq
import base64
import sys
import csv
import resource
import psutil
# mutable parameters
port1 = '5005'
port2 = '5013'
port4 = '5008' #5008
port5 = '5010'
port6 = '5011'
port7='5050'

FPS = 30
maxy=1
miny=-1

#Creating ZMQ Publisher sockets
def pubSocket(port1,port2,port7):
    context = zmq.Context()
    socket1= context.socket(zmq.PUB)
    socket1.bind("tcp://*:%s"%port1)
    socket2= context.socket(zmq.PUB)
    socket2.bind("tcp://*:%s"%port2)
    socket6= context.socket(zmq.PUB)
    socket6.bind("tcp://*:%s"%port7)
    return socket1,socket2,socket6

#Creating ZMQ Subscriber sockets
def subSocket(port4,port5,port6):
    context = zmq.Context()
    socket3= context.socket(zmq.SUB)
    socket3.connect("tcp://localhost:%s" % port4)
    socket4= context.socket(zmq.SUB)
    socket4.connect("tcp://localhost:%s" % port5)
    socket5= context.socket(zmq.SUB)
    socket5.connect("tcp://localhost:%s" % port6)
    return socket3,socket4,socket5

#Function to increase brightness.
#Train data value: 30, 50. Test data value: 70
def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


if __name__ == '__main__':

    socket1,socket2,socket6 = pubSocket(port1,port2,port7)#Publisher sockets
    socket3,socket4,socket5 = subSocket(port4,port5,port6)#subscriber sockets

    # Create environment
    env = CarEnv()

    # For agent speed measurements - keeps last 60 frametimes
    fps_counter = deque(maxlen=30)

    while True:# Loop over episodes
        print('Restarting episode')
        # Reset environment and get initial state
        current_state = env.reset()
        env.Weather_Condition(c=0.0,p=0.0,s=100.0)
        env.collision_hist = []
        socket3.setsockopt(zmq.SUBSCRIBE, b'')
        time_count=[]
        i=0
        while True:# Loop over steps
            try:
                i+=1
                val=[]
                step_start = time.time()
                if(i>120):# For changing scene information midway a scene
                    env.Weather_Condition(c=0.0,p=100.0,s=100.0)
                    #current_state = increase_brightness(current_state, value=70)
                # Show current frame
                print("received image")
                cv2.imshow('Agent', current_state)
                cv2.waitKey(1)
                # For changing scene information midway a scene
                if(i>120):
                    #env.Weather_Condition(c=0.0,p=100.0,s=100.0)
                    current_state = increase_brightness(current_state, value=70)
                image = cv2.imencode('.png', current_state)[1].tostring()
                socket1.send(base64.b64encode(image))
                steer = socket3.recv_string()
                steer = float(steer)
                # if(i>120 and i < 150):
                #     steer+=0.1
                # if(i>165):
                #     #vehicle.set_autopilot(True)
                #     #steer = vehicle.get_control().steer
                #     steer=0.0
                #print(float(steer))
                #current_state.save_to_disk('dataset/videos/mix-video/frame%d'% (l))#save images
                socket6.send_string(str(steer)) #send steering value for plotting
                new_state = env.step(float(steer)) #send in predicted steer value
                current_state = new_state #get new image

                frame_time = time.time() - step_start
                #time_count.append(frame_time)
                fps_counter.append(frame_time)
                cpu = psutil.cpu_percent()#cpu utilization stats
                mem = psutil.virtual_memory()#virtual memory stats
                #val.append(cpu)
                val.append(frame_time)
                #csvfile to store total inference time
                with open('total-inference-time.csv', 'a') as file:
                    writer = csv.writer(file)
                    writer.writerow(val)
                #print("Agent:%f steer:%f" %(len(fps_counter)/sum(fps_counter), float(steer)))

            except KeyboardInterrupt:
                for actor in env.actor_list:
                    actor.destroy()
                time.sleep(4)
                socket1.close()
                socket2.close()
                socket3.close()
                sys.exit()
                print('done.')
