#!/usr/bin/env python3
import numpy as np
from glob import glob
import cv2
import h5py
import os
import random 
import torch
import torch.utils.data as data
ONLY_FRAME_LENS = False
'''
	This script is a demo of feature abstraction using optical flow operations.

	Follow the nuScenes's tutorial (https://www.nuscenes.org/) to extract the mini set. 
	Only images from the CAM_FRONT channel are sued.

	There are 10 video scenes in the v1.0-mini version. 
	All frames (static PNG images) extracted from one video scene shall be placed in one scene folder. 
	Place all scene folders in one folder, e.g. data/nuscenes-v1.0-mini/ 

	Each scene will be splitted into a test (1-48 frames) and a train (49-last frames) feature file in hdf5 format

'''

class FeatureAbstraction:
	def __init__(self, trainroot, testroot, dstroot):

		# Specify output (aka destination) root
		self.dstroot = dstroot
		# Resize inputs
		self.newdim = (320, 240)
		train_folders = [6, 20, 17, 7, 30, 8, 13, 27, 5, 26, 31, 21, 32, 3, 10, 19, 1, 24, 4, 2]
		frame_lens = {}

		for phase in ["train", "in", "out_foggy", "out_night", "out_snowy", "out_rainy", "out_replay"]:
			# List file for train and test loaders
			self.store = []
			frame_lens[phase] = []
			if "train" in phase:
				path = trainroot
				phase_type = "train"
				locs = []
				names = []
				for folder_number in train_folders:
					if folder_number <= 10:
						locs.append(trainroot+"setting_1/"+str(folder_number))
						names.append(trainroot+"setting_1/"+str(folder_number))
					elif folder_number >= 11 and folder_number <= 21:
						locs.append(trainroot+"setting_2/"+str(folder_number-11))
						names.append(trainroot+"setting_2/"+str(folder_number))
					elif folder_number >= 22 and folder_number <= 32:
						locs.append(trainroot+"setting_3/"+str(folder_number-22))
						names.append(trainroot+"setting_3/"+str(folder_number))
			elif ("in" in phase) and ("rainy" not in phase):
				path = testroot+phase+"/"
				locs = glob(path + "*")
				names = locs
				phase_type = "test"
			elif "out" in phase:
				path = testroot+phase+"/out/"
				locs = glob(path + "*")
				names = locs
				phase_type = "test"
			
			for idx, scenefolder in enumerate(locs):
				frames = []
				# It is time series. Frame order matters !!!
				for imagefile in sorted(glob(scenefolder + "/*.png")):
					frames.append(imagefile)

				# Fetch the 1st frame
				frame_lens[phase].append(len(frames))
				if ONLY_FRAME_LENS: continue
				im1 = cv2.cvtColor(cv2.resize(cv2.imread(frames[0]), self.newdim), cv2.COLOR_BGR2GRAY)
				features_x, features_y = [], []
				im2, flow = None, None
				for i in range(1, len(frames)):
					if im2 is None:
						im2 = cv2.cvtColor(cv2.resize(cv2.imread(frames[i]), self.newdim), cv2.COLOR_BGR2GRAY)
					else:
						im1 = im2
						im2 = cv2.cvtColor(cv2.resize(cv2.imread(frames[i]), self.newdim), cv2.COLOR_BGR2GRAY)

					flow = cv2.calcOpticalFlowFarneback(im1, im2, flow, 
							pyr_scale = 0.5, levels = 1, iterations = 1, 
							winsize = 11, poly_n = 5, poly_sigma = 1.1,  
							flags = 0 if flow is None else cv2.OPTFLOW_USE_INITIAL_FLOW )
					features_x.append(cv2.resize(flow[..., 0], None, fx=0.5, fy=0.5))
					features_y.append(cv2.resize(flow[..., 1], None, fx=0.5, fy=0.5))

				# Write optic flow fields of one video episode to a h5 file
				file = self.dstroot + phase + "." + names[idx].split("/")[-1] + ".h5"
				with h5py.File(file, "w") as f:
					f.create_dataset("x", data=features_x)
					f.create_dataset("y", data=features_y)
				self.store.append(file)

			if ONLY_FRAME_LENS: 
				continue
			h5fillist= self.dstroot + phase+"."+phase_type
			with open(h5fillist, "w") as f:
				for scene in self.store: 
					f.write(scene+"\n")

			print("See feature extraction results for phase={} in {}".format(phase, h5fillist))
		print('frame lengths for phases are: \n', frame_lens)

if __name__ == "__main__":
	trainroot = '../carla_data/training/' 
	testroot = '../carla_data/testing/'
	dstroot = '../carla_features_all/'

	try:
		os.mkdir(dstroot)
	except:
		pass
	FeatureAbstraction(trainroot, testroot, dstroot)


