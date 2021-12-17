#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import time

vx = [0]
vy = [0]
sx = [0]
sy = [0]
w1 = [0]
w2 = [0]
wy = [0]
px = [0]
py = [0]
mx = [0]
my = [0]

# MUTABLE
truncate = True
MAXARRAYSIZE = 20

def truncateGraph():
    global vx,vy,sx,sy,w1,w2,wy,px,py,mx,my
    if (len(vx)>MAXARRAYSIZE):
        vx= vx[len(vx)-MAXARRAYSIZE:len(vx)]
    if (len(vy)>MAXARRAYSIZE):
        vy= vy[len(vy)-MAXARRAYSIZE:len(vy)]
    if (len(sx)>MAXARRAYSIZE):
        sx= sx[len(sx)-MAXARRAYSIZE:len(sx)]
    if (len(sy)>MAXARRAYSIZE):
        sy = sy[len(sy)-MAXARRAYSIZE:len(sy)]
    if (len(px)>MAXARRAYSIZE):
        px= px[len(px)-MAXARRAYSIZE:len(px)]
    if (len(py)>MAXARRAYSIZE):
        py= py[len(py)-MAXARRAYSIZE:len(py)]
    if (len(mx)>MAXARRAYSIZE):
        mx= mx[len(mx)-MAXARRAYSIZE:len(mx)]
    if (len(my)>MAXARRAYSIZE):
        my = my[len(my)-MAXARRAYSIZE:len(my)]
    if (len(w1)>MAXARRAYSIZE):
        w1= w1[len(w1)-MAXARRAYSIZE:len(w1)]
    if (len(w2)>MAXARRAYSIZE):
        w2 = w2[len(w2)-MAXARRAYSIZE:len(w2)]
    if (len(wy)>MAXARRAYSIZE):
        wy = wy[len(wy)-MAXARRAYSIZE:len(wy)]

def addfeature1pval(xn,yn):
    global vx,vy
    vx.append(xn)
    vy.append(yn)

def addfeature1Mval(xn,yn):
    global sx,sy
    sx.append(xn)
    sy.append(yn)

def addfeature2pval(xn,yn):
    global px,py
    px.append(xn)
    py.append(yn)

def addfeature2Mval(xn,yn):
    global mx,my
    mx.append(xn)
    my.append(yn)

def addAnomaly(xn,x1n):
    global w1,w2,wy
    wy.append(xn)
    w1.append(x1n)
    #w2.append(yn)

def getfeature1pval():
    global vx,vy
    return vx,vy

def getfeature1Mval():
    global sx,sy
    return sx,sy

def getfeature2pval():
    global px,py
    return px,py

def getfeature2Mval():
    global mx,my
    return mx,my

def getAnomaly():
    global wx,wy
    return wy,w1

style.use('ggplot')

def animateGraph():
    global fig
    fig = plt.figure(figsize=(6,4))
    ax3 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)

    def animate(i):
        vx,vy = getfeature1pval()
        sx,sy = getfeature1Mval()
        wy,w1 = getAnomaly()
        a,p = getfeature2pval()
        a,m = getfeature2Mval()

        try:
            ax3.clear()
            ax3.set_ylim(0,1.1)
            ax3.plot(wy,w1,label = "Anomaly", color='blue')
            ax3.set_title("Detector")
            ax3.set_ylabel("detection")
            ax3.set_xlabel("Time(s)")
            ax3.autoscale(enable=False, axis='y')
            ax2.clear()
            ax2.set_ylim(-0.25,0.25)
            ax2.set_yticks([-0.25,-0.1,0,0.1,0.25])
            ax2.plot(vx,vy,color='red')
            ax2.set_ylabel("Steer")
            ax2.set_xlabel("Time(s)")
            ax2.autoscale(enable=False, axis='y')
        except:
            print('s')
    plt.grid()
    plt.subplots_adjust(hspace = 1.5,wspace = 1.5)
    ani = animation.FuncAnimation(fig, animate, interval=500)
    plt.show()
