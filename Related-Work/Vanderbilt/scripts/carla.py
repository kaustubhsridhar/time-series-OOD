import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from glob import glob
from PIL import Image

class CARLADataset(Dataset):
    def __init__(self, root_dir, new_size, transform=None):
        """
        Args:
            root_dir (string): folder containing settings
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.size = new_size

        train_folders = [6, 20, 17, 7, 30, 8, 13, 27, 5, 26, 31, 21, 32, 3, 10, 19, 1, 24, 4, 2]
        folder_locs = []
        for folder_number in train_folders:
            if folder_number <= 10:
                folder_locs.append(self.root_dir+"setting_1/"+str(folder_number))
            elif folder_number >= 11 and folder_number <= 21:
                folder_locs.append(self.root_dir+"setting_2/"+str(folder_number-11))
            elif folder_number >= 22 and folder_number <= 32:
                folder_locs.append(self.root_dir+"setting_3/"+str(folder_number-22))
    
        self.img_locs = []
        for idx, scenefolder in enumerate(folder_locs):
            for imagefile in sorted(glob(scenefolder + "/*.png")):
                self.img_locs.append(imagefile)

    def __len__(self):
        return len(self.img_locs)

    def __getitem__(self, idx):
        img_loc = self.img_locs[idx]
        img = Image.open(img_loc)
        img = img.resize(self.size)
        img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        # print(img.shape)

        return (img, 1) # labels are unimportant

class CARLA_test_Dataset(Dataset):
    def __init__(self, OOD_dir, iD_dir, new_size, transform=None):
        """
        Args:
            root_dir (string): folder containing settings
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.size = new_size

        OOD_locs = glob(OOD_dir + "*")
        iD_locs = glob(iD_dir + "*")
        self.img_locs_and_GTs = []
        for idx, scenefolder in enumerate(OOD_locs):
            for imagefile in sorted(glob(scenefolder + "/*.png")):
                self.img_locs_and_GTs.append((imagefile, 0))

        for idx, scenefolder in enumerate(iD_locs):
            for imagefile in sorted(glob(scenefolder + "/*.png")):
                self.img_locs_and_GTs.append((imagefile, 1))

    def __len__(self):
        return len(self.img_locs_and_GTs)

    def __getitem__(self, idx):
        img_loc, is_iD = self.img_locs_and_GTs[idx]
        img = Image.open(img_loc)
        img = img.resize(self.size)
        img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        # print(img.shape)

        return (img, is_iD) # labels are imp here


class CARLA_calib_Dataset(Dataset):
    def __init__(self, root_dir, new_size, transform=None):
        """
        Args:
            root_dir (string): folder containing settings
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.size = new_size

        train_folders = [12, 15, 14, 2, 16, 18, 0, 25, 9, 23, 28, 22, 11]
        folder_locs = []
        for folder_number in train_folders:
            if folder_number <= 10:
                folder_locs.append(self.root_dir+"setting_1/"+str(folder_number))
            elif folder_number >= 11 and folder_number <= 21:
                folder_locs.append(self.root_dir+"setting_2/"+str(folder_number-11))
            elif folder_number >= 22 and folder_number <= 32:
                folder_locs.append(self.root_dir+"setting_3/"+str(folder_number-22))
    
        self.img_locs = []
        for idx, scenefolder in enumerate(folder_locs):
            for imagefile in sorted(glob(scenefolder + "/*.png")):
                self.img_locs.append(imagefile)

    def __len__(self):
        return len(self.img_locs)

    def __getitem__(self, idx):
        img_loc = self.img_locs[idx]
        img = Image.open(img_loc)
        img = img.resize(self.size)
        img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)

        return (img, 1) # labels are unimportant
