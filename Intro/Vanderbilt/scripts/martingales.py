import scipy.integrate as integrate
from scipy import stats
import numpy as np
import queue
import random

class RPM(object):
    def __init__(self, epsilon,sliding_window_size=None):
        self.M = 1.0
        self.epsilon = epsilon
        self.sliding_window_size = sliding_window_size
        if self.sliding_window_size:
            self.betting_queue = queue.Queue(self.sliding_window_size)

    
    def __call__(self, p):
        if (p <= 0.005):
            p = 0.005
        betting = self.epsilon *  (p ** (self.epsilon-1.0))
        if self.sliding_window_size:
            if self.betting_queue.full():
                self.M /= self.betting_queue.get()
            self.betting_queue.put(betting)

        self.M *= betting
        return self.M


class SMM(object):
    def __init__(self, sliding_window_size=None):
        self.p_list = []
        self.sliding_window_size = sliding_window_size
    
    def __integrand(self, x):
        result = 1.0
        for i in range(len(self.p_list)):
            result *= x*(self.p_list[i]**(x-1.0))
        return result
    
    def __call__(self, p):
        if (p<=0.005):
            p = 0.005
        self.p_list.append(p)
        if self.sliding_window_size:
            if len(self.p_list)>=self.sliding_window_size:
                self.p_list = self.p_list[-1 * self.sliding_window_size:]
        M, _ = integrate.quad(self.__integrand, 0.0, 1.0)
        return M