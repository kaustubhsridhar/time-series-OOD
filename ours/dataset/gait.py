from __future__ import print_function
from PIL import Image, PILLOW_VERSION
import os
import os.path
import numpy as np
import sys
import numbers
import random
from torchvision import transforms
import cv2 
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch
import torch.utils.data as data
import csv

import copy

from scipy.ndimage import maximum_filter1d
from scipy.ndimage import minimum_filter1d


import matplotlib.pyplot as plt  

class GAIT(data.Dataset):
    """Drift dataset
    Args:
        root_dir (string): Directory with training/test data.
        train (bool): train split or test split.
        win_len (int): number of frames in win.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self,
                root_dir,
                win_len=16,
                train=True,
                cal=False,
                in_dist_test=False,
                transformation_list = ["low_pass", 'high_low', 'low_high', 'identity']):
        
        self.root_dir = root_dir
        self.win_total_datapoints = win_len
        self.train = train
        self.cal = cal
        self.in_dist_test = in_dist_test
        self.traces = []
        self.tranformation_list = transformation_list
        self.num_classes = len(self.tranformation_list)

        if self.train:

            #print("Training")
            no_traces = 6 
            training_traces = [1,2,3,4,5,6] 

            for training_trace_id in training_traces: 
                f = open("{}/control{}.ts".format(root_dir,training_trace_id), "r") 
                cur_trace_data = [] 
                while(True): 
                    line = f.readline() 
                    if not line: 
                        break  
                    data = [float(item) for item in line.split()]  
                    data = data[1:] # excluding time data  
                    cur_trace_data.append(data) 
                f.close() 
     
                self.traces.append(cur_trace_data) # 3D

        elif self.cal:
            no_traces = 5 
            cal_traces = [7,8,9,10,11] 

            for cal_trace_id in cal_traces: 
                f = open("{}/control{}.ts".format(root_dir,cal_trace_id), "r") 
                cur_trace_data = [] 
                while(True): 
                    line = f.readline() 
                    if not line: 
                        break  
                    data = [float(item) for item in line.split()]  
                    data = data[1:] # excluding time data  
                    cur_trace_data.append(data) 
                f.close() 
     
                self.traces.append(cur_trace_data) # 3D

        else:

            if self.in_dist_test==True: # Testing for in-distribution 
                no_traces = 5 
                id_test_traces = [12,13,14,15,16] 

                for id_test_trace_id in id_test_traces: 
                    f = open("{}/control{}.ts".format(root_dir,id_test_trace_id), "r") 
                    cur_trace_data = [] 
                    while(True): 
                        line = f.readline() 
                        if not line: 
                            break  
                        data = [float(item) for item in line.split()]  
                        data = data[1:] # excluding time data  
                        cur_trace_data.append(data) 
                    f.close()

                    self.traces.append(cur_trace_data) # 3D


            else:
                no_traces = 48
                ood_types = ['park', 'als', 'hunt'] # 82 on park, 59 on als, 90 on hunt, 86 on both park and hunt
                no_ood_traces = [15, 13, 20]

                # severe group
                trace_ids = {'park': [1,4,7,8,10,11,12,13,14], 
                'als' : [1,12,13,4,5,6,7,3,9],
                'hunt' : [3,4,7,10,13,15,16,18,19]} 

                ood_types = ['als']

                print(ood_types)

                for i,ood_type in enumerate(ood_types): # enumerating on the ood_type
                    # for ood_trace_id in range(1,no_ood_traces[i]+1): 
                    for ood_trace_id in trace_ids[ood_type]: 
                            f = open("{}/{}{}.ts".format(root_dir,ood_type,ood_trace_id), "r") 
                            cur_trace_data = [] 
                            while(True): 
                                line = f.readline() 
                                if not line: 
                                    break  
                                data = [float(item) for item in line.split()]  
                                data = data[1:] # excluding time data  
                                cur_trace_data.append(data) 
                            f.close()

                            self.traces.append(cur_trace_data) # 3D   
    
    def __len__(self):
        return len(self.traces)

    def transform_win(self, win): # win is a 2D list

        trans_win = torch.FloatTensor(win) # trans_win becomes a 2D tensor of win_len X 12 (sensor measurements for 1 time step)
        trans_win = trans_win.unsqueeze(0)  # trans_win is now a 3D tensor. 1 (no. of channels for conv) X win_len X 12

        return trans_win

    def apply_filter_on_2D_data(self, input_data, filter_coeffs): # input_data is 2D
        input_data = np.array(input_data) 
        num_cols = len(input_data[0])
        num_rows = len(input_data)

        output = []
        for c in range(num_cols):
            output.append(np.convolve(input_data[:,c], filter_coeffs,'valid'))
        
        output = np.array(output)
        output = output.transpose()

        return output
    
    def apply_periodic_filter_on_2D_data(self, input_data, filter1_coeffs, filter2_coeffs): # input_data is 2D
        input_data = np.array(input_data) 
        num_cols = len(input_data[0])
        num_rows = len(input_data)

        output = []
        for c in range(num_cols//2):
            output.append(np.convolve(input_data[:,c], filter1_coeffs,'valid'))
        
        for c in range(num_cols//2, num_cols):
            output.append(np.convolve(input_data[:,c], filter2_coeffs,'valid'))

        output = np.array(output)
        output = output.transpose()

        return output
  

    def __getitem__(self, idx): # this is only for CAL/TRAIN
        """
        Returns:
            original win data, transformed win data and the applied transformation
        """

        tracedata = self.traces[idx] # tracedata = 2D list : time steps X data

        length = len(tracedata) # no. of time steps

        win_start = random.randint(0, length - self.win_total_datapoints)

        orig_win =  tracedata[win_start:win_start+self.win_total_datapoints] # 2D list
        trans_win = tracedata[win_start:win_start+self.win_total_datapoints] # 2D list

        if self.tranformation_list[transform_id] == "low_pass":
            trans_win = self.apply_filter_on_2D_data(trans_win, [1/3,1/3,1/3])
        
        elif self.tranformation_list[transform_id] == "high_pass":
            trans_win = self.apply_filter_on_2D_data(trans_win,  [-1/2, 0, 1/2])

        elif self.tranformation_list[transform_id] == "low_high": # low-pass on half, high pass on half
            trans_win = self.apply_periodic_filter_on_2D_data(trans_win, filter1_coeffs=[1/3,1/3,1/3], filter2_coeffs=[-1/2, 0, 1/2])
        
        elif self.tranformation_list[transform_id] == "high_low": # low-pass on half, high pass on half
            trans_win = self.apply_periodic_filter_on_2D_data(trans_win, filter1_coeffs=[-1/2, 0, 1/2], filter2_coeffs=[1/3,1/3,1/3])

        elif self.tranformation_list[transform_id] == "identity": #identity
            pass

        else:
            raise Exception("Invalid transformation")
        

        # converting to tensors and new dim = 1 (no. of channels for conv) X win_len X 12
        trans_win = self.transform_win(trans_win)

        orig_win = self.transform_win(orig_win)
        orig_win = orig_win[:,1:-1,:]

        if self.tranformation_list[transform_id] == "identity":
            trans_win = trans_win[:,1:-1,:]

        return orig_win, trans_win, transform_id

    def __get_test_item__(self, idx): # generator for getting sequential shuffled tuples/windows on test data

        tracedata = self.traces[idx]
    
        length = len(tracedata)

        # last_win_starting_point = max(length-(2*self.win_total_datapoints),1) # to get at least one datapoint from the trace if the trace is too short (total len = 2*win_total_datapoints)

        for i in range(0, length-(1*self.win_total_datapoints)+1): 
            win_start = i

            transform_id = random.randint(0, self.num_classes-1)
            
            orig_win =  tracedata[win_start:win_start+self.win_total_datapoints]
            trans_win = tracedata[win_start:win_start+self.win_total_datapoints]

            if self.tranformation_list[transform_id] == "low_pass":
                trans_win = self.apply_filter_on_2D_data(trans_win, [1/3,1/3,1/3])
            
            elif self.tranformation_list[transform_id] == "high_pass":
                trans_win = self.apply_filter_on_2D_data(trans_win,  [-1/2, 0, 1/2])
            
            elif self.tranformation_list[transform_id] == "low_high": # low-pass on half, high pass on half
                trans_win = self.apply_periodic_filter_on_2D_data(trans_win, filter1_coeffs=[1/3,1/3,1/3], filter2_coeffs=[-1/2, 0, 1/2])
            
            elif self.tranformation_list[transform_id] == "high_low": # low-pass on half, high pass on half
                trans_win = self.apply_periodic_filter_on_2D_data(trans_win, filter1_coeffs=[-1/2, 0, 1/2], filter2_coeffs=[1/3,1/3,1/3])

            elif self.tranformation_list[transform_id] == "identity": #identity
                pass

            else:
                raise Exception("Invalid transformation")
            

            # converting to tensors and new dim = 1 (no. of channels for conv) X win_len X 12
            trans_win = self.transform_win(trans_win)

            orig_win = self.transform_win(orig_win)
            orig_win = orig_win[:,1:-1,:]

            if self.tranformation_list[transform_id] == "identity":
                trans_win = trans_win[:,1:-1,:]

            yield orig_win, trans_win, transform_id
