#!/usr/bin/env python3

import time 
import logging
import os
import random
import torch
import torch.utils.data
from utils.diff_utils import * 

import pandas as pd 
import numpy as np
import csv, json

from tqdm import tqdm

class ModulationLoader(torch.utils.data.Dataset):
    def __init__(self, data_path, gs_path=None):
        super().__init__()

        self.conditional = gs_path is not None 
        
        self.modulations = self.unconditional_load_modulations(data_path)
        
        print("data shape, dataset len: ", self.modulations[0].shape, len(self.modulations))

            
        
    def __len__(self):
        return len(self.modulations)

    def __getitem__(self, index):

        cond = self.conditions[index] if self.conditional else False
        return {
            "cond" : cond,
            "latent" : self.modulations[index]         
        }
    
    def unconditional_load_modulations(self, data_source, f_name="latent.txt", add_flip_augment=False):
        files = []
        length = len(os.listdir(data_source))
        for idx in range(length):
            instance_filename = os.path.join(data_source, str(idx), f_name)
            files.append( torch.from_numpy(np.loadtxt(instance_filename)).float() )
        return files
    