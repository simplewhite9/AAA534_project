import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

import copy, json, pickle
import glob, os
import h5py
import numpy as np
import pandas as pd
from collections import OrderedDict



class ModelNet40(Dataset):

    def __init__(self, args = None, tokenizer = None, split='train'):

        self.dataset_dir = f'../data/modelnet40_ply_hdf5_2048/'
        text_file = os.path.join(self.dataset_dir, 'shape_names.txt')
        self.classnames = self.read_classnames(text_file)

        data, label = self.load_data(os.path.join(self.dataset_dir, f'{split}_files.txt'))
        self.prompt = "A three-dimensional model of an xxxxxx composed of gray, fuzzy balls."
        self.tokenizer = tokenizer

        self.data = data
        self.label = label


    def load_data(self, data_path):
        all_data = []
        all_label = []
        with open(data_path, "r") as f:
            for h5_name in f.readlines():
                f = h5py.File(h5_name.strip(), 'r')
                data = f['data'][:].astype('float32')
                label = f['label'][:].astype('int64')
                f.close()
                all_data.append(data)
                all_label.append(label)
        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        
        return all_data, all_label
        

    @staticmethod
    def read_classnames(text_file):
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
    

    def __getitem__(self, idx):
        data = torch.from_numpy(self.data[idx])
        label = (self.label[idx]).item()
        classname = self.classnames[label]
        text_id = self.tokenizer(self.prompt.replace('xxxxx', classname))

        return {"data": data, "text_id":text_id, "label": label, "classname": classname}


    def __len__(self):
        return len(self.data)


def modelnet40_collate(batch):
    bs = len(batch)
    data = torch.stack([batch[i]["data"] for i in range(bs)])
    text_id = torch.stack([batch[i]["text_id"] for i in range(bs)])
    
    label = [batch[i]["label"] for i in range(bs)]
    classname = [batch[i]["classname"] for i in range(bs)]

    return {"data": data, "text_id":text_id, "label": label, "classname": classname}
