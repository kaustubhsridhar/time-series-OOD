#
# Code written by : Souradeep Dutta,
#  duttaso@seas.upenn.edu, souradeep.dutta@colorado.edu
# Website : https://sites.google.com/site/duttasouradeep39/
#

import os
import json
import numpy as np
from shutil import copyfile
import copy
import random
import torch
import time
import sys
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from copy import deepcopy as dc

from distance_calculations.find_features import return_feature_vector
from distance_calculations.pytorch_modified_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from memories.data import data
from memories.carla_data import carla_data
import re


class memory:

    def __init__(self, device):
        self.device = device

        self.data_point = carla_data(self.device)

        # This doesn't need to be saved to disk, but is a good book-keeping data-value
        self.data_point_filename = None

        # This is the acceptance radii sort of thing
        self.distance_score = []

        # This is the list of samples it thinks is within its reach from the prototype point
        self.samples_list = []

        # The similarity score for this memory
        self.distance_score_file = "distance_score.json"

        # The list of lesions it thinks is under the belt
        self.datapoints_solved_filename = "content.json"

        self.memory_distance_file = "other_memory_distance.json"

        self.weight = 0


    def create_memory_from_files(self, files_list):

        self.data_point.create_data_from_scan(files_list)
        self.samples_list.append(files_list)


    def read_memory(self, dir_name):

        assert os.path.exists(dir_name)

        self.data_point.read_data_as_memory(dir_name)


        fp = open(os.path.join(dir_name, self.datapoints_solved_filename), "r")
        self.samples_list = json.load(fp)
        fp.close()
        self.weight = len(self.samples_list)

        fp = open(os.path.join(dir_name, self.distance_score_file), "r")
        self.distance_score = json.load(fp)
        fp.close()
    
    def read_other_distances(self,dir_name):
        assert os.path.exists(dir_name)
        fp = open(os.path.join(dir_name, self.memory_distance_file), "r")
        other_distance_list = json.load(fp)
        fp.close()
        return other_distance_list


    def save_memory(self, dir_name):

        if not os.path.exists(dir_name):
            os.mkdir(dir_name)


        self.data_point.save_data_as_memory(dir_name)

        # Saving the admitted/solved images under the current memory
        fp = open(os.path.join(dir_name, self.datapoints_solved_filename), "w")
        json.dump(self.samples_list, fp)
        fp.close()

        # Distance score for this memory
        fp = open(os.path.join(dir_name, self.distance_score_file), "w")
        json.dump(self.distance_score, fp)
        fp.close()
    
    def save_distance_other_memories(self, dir_name,distance_list):

        # Saving the admitted/solved images under the current memory
        fp = open(os.path.join(dir_name, self.memory_distance_file), "w")
        json.dump(distance_list, fp)
        fp.close()


    def apply_memory(self, all_data_collection, unsolved_set):


        all_distances = self.data_point.compute_distance_batched(all_data_collection)
        count = 0
        for name in all_distances.keys():

            if all_distances[name] < self.distance_score[0]:
                
                self.samples_list.append(all_data_collection[name]["files"])

                if name in unsolved_set :
                    count += 1
                    unsolved_set.remove(name)
                else:
                    pass

        return unsolved_set, all_distances
