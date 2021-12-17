#!/usr/bin/env python3

from threading import Thread
import threading
import plotTools-simplex-change
import zmq

port1 = '5020'
port2 = '5021'
port3 = '5050'

def subSocket(port1,port2,port3):
    context = zmq.Context()
    socket1 = context.socket(zmq.SUB)
    socket1.connect("tcp://localhost:%s" % port1)
    socket2 = context.socket(zmq.SUB)
    socket2.connect("tcp://localhost:%s" % port2)
    socket3 = context.socket(zmq.SUB)
    socket3.connect("tcp://localhost:%s" % port3)
    return socket1,socket2,socket3

def plotVelocity():
    print('animating')
    plotTools.animateGraph()

def plotter(socket1,socket2,socket3):
    socket1.setsockopt(zmq.SUBSCRIBE, b'')
    socket2.setsockopt(zmq.SUBSCRIBE, b'')
    socket3.setsockopt(zmq.SUBSCRIBE, b'')
    plotThread = Thread(target = plotVelocity)
    plotThread.daemon = True
    plotThread.start()
    while True:
        data = socket1.recv_string()
        print(data)
        data1 = socket2.recv_string()
        data2 = socket3.recv_string()
        steer = float(data2)
        i,final_pval,m_val,val = data.split()
        i,final_pval1,m_val1,val1 = data1.split()
        i=float(i)*0.1
        # if(i>1.3):
        #     val=1
        plotTools.addAnomaly(i,val1)   #add the detector results
        plotTools.addfeature1pval(i,steer) #add feature1 pval
        plotTools.addfeature1Mval(i,m_val) #add feature1 mval
        plotTools.addfeature2pval(i,final_pval1)#add feature2 pval
        plotTools.addfeature2Mval(i,m_val1)#add feature2 mval
    plotThread.join()
    socket1.close()
    socket2.close()

if __name__ == '__main__':
    socket1,socket2,socket3 = subSocket(port1,port2,port3)
    plotter(socket1,socket2,socket3)
