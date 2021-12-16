from __future__ import print_function
from PIL import Image, PILLOW_VERSION
import os
import os.path
import numpy as np
import sys
import numbers
import random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import torch
import torch.utils.data as data
import csv

import copy

import matplotlib.pyplot as plt 

class DriftDataset(data.Dataset):
    """Drift dataset

    Args:
        root_dir (string): Directory with training/test data.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip.
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self, root_dir, clip_len=16, train=True, cal=False, transforms_=None, img_hgt=360, img_width=640, in_dist_test=False):
        self.root_dir = root_dir
        self.clip_total_frames = clip_len
        self.train = train
        self.cal = cal
        self.in_dist_test = in_dist_test
        self.transforms_ = transforms_
        self.img_hgt = img_hgt
        self.img_width = img_width
        self.videos = []

        self.cal_speed_transform_idx = None

        if self.train:

            #print("Training")
            no_episodes = 24 #new
            # no_episodes = 14 #old
            
            for episode in range(1,no_episodes+1): 
                #print("episode: ", episode)
                file = open("{}/{}/n_frames".format(root_dir, episode), 'r')
                no_images = int(file.read())
                video_frames = [] 
                for img in range(1,no_images+1):
                    #print("Image: ", img)
                    with Image.open('{}/{}/image_{}.jpg'.format(root_dir, episode, (str(img).zfill(5)))) as im: 
                        video_frames.append(im.resize((self.img_width, self.img_hgt)))  
                self.videos.append(video_frames)

        elif self.cal:
            #print("Calibration")
            no_episodes = 14 #new
            # no_episodes = 9 #old

            self.cal_speed_transform_idx = [0]*no_episodes

            for episode in range(1,no_episodes+1): 
                #print("Episode: ", episode)
                file = open("{}/{}/n_frames".format(root_dir, episode), 'r')
                no_images = int(file.read())
                video_frames = [] 
                for img in range(1,no_images+1):
                    #print("Image: ", img)
                    with Image.open('{}/{}/image_{}.jpg'.format(root_dir, episode, (str(img).zfill(5)))) as im: 
                        video_frames.append(im.resize((self.img_width, self.img_hgt)))  
                self.videos.append(video_frames)

        else:

            if self.in_dist_test==True: # Testing for in-distribution 
                no_episodes = 34 #new
                # no_episodes = 9 #old
                ext = ''
            else:
                no_episodes = 100
                ext='_1'

            for episode in range(1,no_episodes+1):
                file = open("{}/{}{}/n_frames".format(root_dir, episode, ext), 'r')
                no_images = int(file.read())
                video_frames = [] 
                for img in range(1,no_images+1):
                    with Image.open('{}/{}{}/image_{}.jpg'.format(root_dir, episode, ext, (str(img).zfill(5)))) as im:  
                        video_frames.append(im.resize((self.img_width, self.img_hgt)))
                # video_frames = trace -> optical flow trace func
                # self.opt_flow_trace.append()
                self.videos.append(video_frames)      
    
    def __len__(self):
        return len(self.videos)

    def transform_clip(self, clip):
        trans_clip = []
        
        for frame in clip:
            frame = self.transforms_(frame) # tensor [C x H x W]
            trans_clip.append(frame)
        trans_clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
        return trans_clip

    def __getitem__(self, idx): # this is only for CAL/TRAIN
        """
        Returns:
            original video frames, transformed video frames and the applied transformation
        """

        videodata = self.videos[idx]

        length = len(videodata)
        trans_clip = [] # transformed clip
        orig_clip = [] # orig clip
        clip_start = random.randint(0, length - self.clip_total_frames)
        # print("Clip start: ", clip_start)
        for index in range(self.clip_total_frames): 
            orig_clip.append(videodata[clip_start + index]) 
            trans_clip.append(videodata[clip_start + index]) 

        # random transformation selection with   -0: 2x speed shuffle, 1: reverse, 2: periodic (forward, backward)
        transform_id = random.randint(0, 3)
        # print("clip_start: ", clip_start)
        # print("Before transform: {} on idx: {} with transformation: {}".format(self.cal_speed_transform_idx, idx, transform_id))

        if transform_id == 0: # prev 2x speed - periodic (backward, forward)
            
            if self.cal and self.clip_total_frames==24: # sanity check to make sure that the speed transform is applied only once on the calibration data point because the starting point will always be zero if clip_len = 24 in case of speed transform as there are only 49 frames in the calibration data points
                if self.cal_speed_transform_idx[idx] == 1: # speed transform has been applied to this cal data point already
                    transform_id = random.randint(1, 3)
                else: # speed transform has not been applied to this data point, so it can be applied now
                   orig_clip = []
                   trans_clip = []
                   clip_start = random.randint(0, length - (2*self.clip_total_frames)) # multiplying 2 for 2x speed
                   # print("Clip start: ", clip_start)
                   for index in range(self.clip_total_frames):
                       orig_clip.append(videodata[clip_start + index])

                   for index in range(self.clip_total_frames):
                       trans_clip.append(videodata[clip_start + 2*index])
                   self.cal_speed_transform_idx[idx] = 1 


        if transform_id == 1: # shuffle
        
            random.shuffle(trans_clip)

        elif transform_id == 2: # reverse

            trans_clip.reverse()

        else: # periodic (forward, backward)
           trans_clip[self.clip_total_frames//2:self.clip_total_frames] = reversed(trans_clip[self.clip_total_frames//2:self.clip_total_frames]) 
        
        trans_clip = self.transform_clip(trans_clip)
        orig_clip = self.transform_clip(orig_clip)

        # print("After transform, cal_speed_transform_idx: {}, and applied transform: {}".format(self.cal_speed_transform_idx, transform_id))
        return orig_clip, trans_clip, transform_id

    def __get_test_item__(self, idx): # generator for getting sequential shuffled tuples/windows on test data

        videodata = self.videos[idx]
    
        length = len(videodata)

        # last_win_starting_point = max(length-(2*self.clip_total_frames),1) # to get at least one datapoint from the trace if the trace is too short (total len = 2*clip_total_frames)

        for i in range(0, length-(2*self.clip_total_frames)+1): 
            clip_start = i

            transform_id = random.randint(0, 3)

            trans_clip = [] # transformed clip
            orig_clip = [] # orig clip

            for index in range(self.clip_total_frames): 
                orig_clip.append(videodata[clip_start + index]) 
                trans_clip.append(videodata[clip_start + index])
            
            if transform_id == 0: 
                
                trans_clip = []

                for index in range(self.clip_total_frames):
                    trans_clip.append(videodata[clip_start + 2*index])
                # trans_clip[0:self.clip_total_frames//2] = reversed(trans_clip[0:self.clip_total_frames//2]) 

            if transform_id == 1: # shuffle

                random.shuffle(trans_clip)

            elif transform_id == 2: # reverse

                trans_clip.reverse()

            else: # periodic (forward, backward)
                trans_clip[self.clip_total_frames//2:self.clip_total_frames] = reversed(trans_clip[self.clip_total_frames//2:self.clip_total_frames]) 
            
            trans_clip = self.transform_clip(trans_clip)
            orig_clip = self.transform_clip(orig_clip)

            yield orig_clip, trans_clip, transform_id
