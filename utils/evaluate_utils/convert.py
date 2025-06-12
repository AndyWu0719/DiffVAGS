import numpy as np
from plyfile import PlyData, PlyElement
import torch
import os
import torch
import random
import numpy as np
import traceback
from multiprocessing import Pool
from fnmatch import fnmatch



def convert(data, path):
    xyz = data[:, :3]
    normals = np.zeros_like(xyz)
    f_dc = data[:, 3:6]
    f_rest = data[:, 6:51]
    opacities = data[:,51:52]
    scale = data[:,52:55]
    rotation = data[:,55:59]

    def construct_list_of_attributes():
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(3):
            l.append('f_dc_{}'.format(i))
        for i in range(45):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(3):
            l.append('scale_{}'.format(i))
        for i in range(4):
            l.append('rot_{}'.format(i))
        return l


    write_path = path
    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(write_path)
    