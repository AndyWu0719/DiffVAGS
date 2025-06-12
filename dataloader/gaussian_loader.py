#!/usr/bin/env python3

import time 
import logging
import os
import random
import torch
import torch.utils.data
from . import base 

import pandas as pd 
import numpy as np
import csv, json

from tqdm import tqdm

class GaussianLoader(torch.utils.data.Dataset):

    def __init__(
        self,
        data_path
    ):
        self.data_path = data_path
        self.path_names = [f for f in os.listdir(data_path)]
        self.path_names = self.path_names

        print(len(self.path_names))

    def __getitem__(self, idx):
        gaussian_file_path = os.path.join(self.data_path, self.path_names[idx], 'gaussian.npy')
        occ_file_path = os.path.join(self.data_path, self.path_names[idx], 'occ.npy')
        
        gaussian = np.load(gaussian_file_path)
        occ = np.load(occ_file_path)
        
        gaussian = gaussian[:, :]
        gs = torch.from_numpy(gaussian) 

        occ_indices = np.random.choice(occ.shape[0], 80000, replace=False)
        occ = occ[occ_indices, :]

        gaussian_indices = np.random.choice(gaussian.shape[0], 16000, replace=False)
        gaussian = gaussian[gaussian_indices, :]

        gaussian[:,52:55] = np.exp(gaussian[:,52:55])
        norm = np.linalg.norm(gaussian[:,55:59], ord=2, axis=-1, keepdims=True)
        gaussian[:,55:59] = gaussian[:,55:59] / norm
        
        gaussian_xyz = torch.from_numpy(gaussian[:, :3])
        gaussian_gt = torch.from_numpy(gaussian[:,3:])
        
        occ_xyz = torch.from_numpy(occ[:,:3])
        occ = torch.from_numpy(occ[:,3:])
        data_dict = {
                    "gaussians":gs.float(),
                    "gaussian_xyz":gaussian_xyz.float(),
                    "gt_gaussian":gaussian_gt.float(),
                    "occ_xyz": occ_xyz.float(),
                    "occ": occ.float()
                    }
        
        return data_dict
        
    def __len__(self):
        return len(self.path_names)



class GaussianTestLoader(torch.utils.data.Dataset):

    def __init__(
        self,
        data_path
    ):
        self.data_path = data_path
        self.path_names = [f for f in os.listdir(data_path)]
        self.path_names = self.path_names[:3000]
        

    def __getitem__(self, idx):
        gaussian_file_path = os.path.join(self.data_path, self.path_names[idx], 'gaussian.npy')
        occ_file_path = os.path.join(self.data_path, self.path_names[idx], 'occ.npy')
        
        gaussian = np.load(gaussian_file_path)
        occ = np.load(occ_file_path)

        gaussian = gaussian[:, :]
        gs = torch.from_numpy(gaussian) 

        gaussian = torch.from_numpy(gaussian)

        gaussian_xyz = gaussian[:, :3]
        gaussian_gt = gaussian[:,3:]

        
        occ_xyz = torch.from_numpy(occ[:,:3])
        occ = torch.from_numpy(occ[:,3:])
        data_dict = {
                    "gaussians":gs.float(),
                    "gaussian_xyz":gaussian_xyz.float(),
                    "gt_gaussian":gaussian_gt.float(),
                    "occ_xyz": occ_xyz.float(),
                    "occ": occ.float()
                    }
        return data_dict
        
    def __len__(self):
        return len(self.path_names)
    