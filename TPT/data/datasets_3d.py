import math
import os

import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import OrderedDict
import PIL
from PIL import Image
import h5py


datasets_3d = ['ModelNet40', 'ScanObjectNN']


class ModelNet40(Dataset):
    """ ModelNet40 dataset """
    def __init__(self, root, mode='train', n_shot=None, transform=None):
        self.dataset_dir = root
        
        self.transform = transform
        
        text_file = os.path.join(self.dataset_dir, 'shape_names.txt')
        classnames = self.read_classnames(text_file)
        
        if mode == 'train':
            data, label = self.load_data(os.path.join(self.dataset_dir, 'train_files.txt'))
            self.data = self.read_data(classnames, data, label)
        elif mode == 'test':
            data, label = self.load_data(os.path.join(self.dataset_dir, 'test_files.txt'))
            self.data = self.read_data(classnames, data, label)

        
    def read_data(self, classnames, datas, labels):
        items = []
        
        for i, data in enumerate(datas):
            label = int(labels[i])
            classname = classnames[label]
            items.append([data, label])
            
        return items
    
    def load_data(self, data_path):
        all_data = []
        all_label = []
        with open(data_path, "r") as f:
            for h5_name in f.readlines():
                f = h5py.File(os.path.join(self.dataset_dir, h5_name.strip().split('/')[-1]), 'r')
                data = f['data'][:].astype('float32')
                label = f['label'][:].astype('int64')
                f.close()
                # sampled_idx = np.random.randint(data.shape[1], size=1024)
                # all_data.append(data[:, sampled_idx, :])
                all_data.append(data)
                all_label.append(label)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        
        return all_data, all_label
    
    def read_classnames(self, text_file):
        """Return a dictionary containing
        key-value pairs of <folder name>: <class name>.
        """
        classnames = OrderedDict()
        with open(text_file, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                classname = line.strip()
                classnames[i] = classname
        return classnames

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]