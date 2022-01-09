import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from glob import glob
from PIL import Image

class driftDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, new_size, transform=None):
        """
        Args:
            root_dir (string): folder containing numbered folders
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.size = new_size
    
        folder_locs = glob(self.root_dir + "*")
        self.img_locs = []
        for idx, scenefolder in enumerate(folder_locs):
            for imagefile in sorted(glob(scenefolder + "/*.jpg")):
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


class drift_test_Dataset(Dataset):
    """Face Landmarks dataset."""

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
            for imagefile in sorted(glob(scenefolder + "/*.jpg")):
                self.img_locs_and_GTs.append((imagefile, 0))

        for idx, scenefolder in enumerate(iD_locs):
            for imagefile in sorted(glob(scenefolder + "/*.jpg")):
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

        return (img, is_iD) # labels are unimportant