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

class CARLAVCOPDataset(data.Dataset):
    """Vanderbilt CARLA dataset for video clip order prediction. Generate clips and permutes them on-the-fly.
    
    Args:
        root_dir (string): Directory with training/test data.
        train (bool): train split or test split.
        clip_len (int): number of frames in clip, 1.
        interval (int): number of frames between clips, 0
        tuple_len (int): number of clips in each tuple, 16. LET US START WITH 16
        transforms_ (object): composed transforms which takes in PIL image and output tensors.
    """
    def __init__(self, root_dir, clip_len=16, train=True, transforms_=None, img_size=224, in_dist_test=False):
        self.root_dir = root_dir
        self.clip_total_frames = clip_len
        self.train = train
        self.transforms_ = transforms_
        self.img_size = img_size

        self.videos = []

        if self.train:
            
            settings = [1,2,3]
            no_episodes = 11 # precipitation level in [0, 10] as iD
            no_images_per_episode_in_settings = [148,130,130]

            for i,setting in enumerate(settings):  
                for episode in range(0,no_episodes): 
                    video_frames = [] 
                    for img in range(0,no_images_per_episode_in_settings[i]):
                        with Image.open('{}/setting_{}/{}/{}.png'.format(root_dir,setting, episode, img)) as im: 
                            video_frames.append(im.resize((self.img_size, self.img_size)))  
                    self.videos.append(video_frames)
        else:

            old_in_episodes = np.array([0,1,3,12,15,24,25,27,28,37,38,46,49,61,62,68,70,71,74,76,78,80,81,82,85,95,97]) # adding 61 here to remove it from OOD set as the OOD trace no. 61 has corrupted images
            total_episodes = np.array([i for i in range(0,100)]) 
            out_episodes = np.setdiff1d(total_episodes,old_in_episodes) # those episodes in which precipitation becomes >= 20 at some point in time

            if in_dist_test==True: # Testing for in-distribution 
                episodes = np.array([i for i in range(0,27)]) # new tst in_episodes with precipitation <=10
                folder_name = 'in'
            else:
                # required for CARLA rainy
                # episodes = out_episodes
                ######################### rainy ends here
                episodes = np.array([i for i in range(0,27)]) # for foggy/night/snowy,new_rainy
                ######################## other OODs ends here
                # folder_name = 'out' # for CARLA rainy
                folder_name = 'out_rainy/out'
                ### TESTING Reverse of test iD as OOD traces
                #episodes = np.array([i for i in range(0,27)]) # new tst in_episodes with precipitation <=10
                #folder_name = 'in'

            print("Episodes: ", episodes)
            print("folder_name: ", folder_name)

            for episode in episodes:
                file = open("{}/{}/{}/label.csv".format(root_dir,folder_name,episode))
                reader = csv.reader(file)
                no_images= len(list(reader)) # no_images = number of frames in the trace/episode
                video_frames = [] 
                # for img in reversed(range(0,no_images)): # for testing reverse OODs
                #print("episode: ", episode)
                for img in range(0,no_images):
                    #print(img)
                    with Image.open("{}/{}/{}/{}.png".format(root_dir,folder_name,episode,img)) as im: 
                        video_frames.append(im.resize((self.img_size, self.img_size)))
                self.videos.append(video_frames)      
    
    def __len__(self):
        return len(self.videos)

    def transform_clip(self, clip):
        trans_clip = []
        
        for frame in clip:
            frame = frame.convert('RGB') # Remove the 4th channel from the CARLA images
            frame = self.transforms_(frame) # tensor [C x H x W]
            trans_clip.append(frame)
        trans_clip = torch.stack(trans_clip).permute([1, 0, 2, 3])
        return trans_clip

    def __getitem__(self, idx): # this is only for CAL/TRAIN
        """
        Returns:
            transformed video frames and the applied transformation
        """

        videodata = self.videos[idx]
    
        length = len(videodata)

        trans_clip = [] # transformed clip
        orig_clip = [] # orig clip
        clip_start = random.randint(0, length - self.clip_total_frames)

        for index in range(self.clip_total_frames): 
            orig_clip.append(videodata[clip_start + index]) 
            trans_clip.append(videodata[clip_start + index]) 

        # random transformation selection with 0: 2x speed, 1: shuffle, 2: reverse, 3: periodic (forward, backward)
        transform_id = random.randint(0, 3)

        if transform_id == 0: # 2x speed
            
            orig_clip = []
            trans_clip = []
            clip_start = random.randint(0, length - (2*self.clip_total_frames)) # multiplying 2 for 2x speed

            for index in range(self.clip_total_frames):
                orig_clip.append(videodata[clip_start + index])

            for index in range(self.clip_total_frames):
                trans_clip.append(videodata[clip_start + 2*index])

        elif transform_id == 1: # shuffle
        
            random.shuffle(trans_clip)

        elif transform_id == 2: # reverse

            trans_clip.reverse()

        else: # periodic (forward, backward)
           trans_clip[self.clip_total_frames//2:self.clip_total_frames] = reversed(trans_clip[self.clip_total_frames//2:self.clip_total_frames]) 
        
        trans_clip = self.transform_clip(trans_clip)
        orig_clip = self.transform_clip(orig_clip)

        return orig_clip, trans_clip, transform_id

    def __get_test_item__(self, idx): # generator for getting sequential shuffled tuples/windows on test data

        videodata = self.videos[idx]
    
        length = len(videodata)

        for i in range(0, length-(2*self.clip_total_frames)+1): # mul 2 to consider speed case
            clip_start = i
            # random transformation selection with 0: 2x speed, 1: shuffle, 2: reverse, 3: periodic (forward, backward)
            transform_id = random.randint(0, 3)

            trans_clip = [] # transformed clip
            orig_clip = [] # orig clip

            for index in range(self.clip_total_frames): 
                orig_clip.append(videodata[clip_start + index]) 
                trans_clip.append(videodata[clip_start + index])
            
            if transform_id == 0: # 2x speed
                
                trans_clip = []

                for index in range(self.clip_total_frames):
                    trans_clip.append(videodata[clip_start + 2*index])

            elif transform_id == 1: # shuffle

                random.shuffle(trans_clip)

            elif transform_id == 2: # reverse

                trans_clip.reverse()

            else: # periodic (forward, backward)
                trans_clip[self.clip_total_frames//2:self.clip_total_frames] = reversed(trans_clip[self.clip_total_frames//2:self.clip_total_frames]) 
            
            trans_clip = self.transform_clip(trans_clip)
            orig_clip = self.transform_clip(orig_clip)

            yield orig_clip, trans_clip, transform_id