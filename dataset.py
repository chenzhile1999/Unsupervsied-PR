import os
from os.path import join
import torch
from torch.utils.data.dataset import Dataset

class CDPdatasets(Dataset):
    def __init__(self, measurements_dir):
        super(CDPdatasets).__init__()
        self.measurements_dir = measurements_dir
        self.fns = [data for data in os.listdir(self.measurements_dir) if data.endswith(".pt")]
        self.size = len(self.fns)

    def __getitem__(self, index):
        y_path = join(self.measurements_dir, self.fns[index])
        data = torch.load(y_path)
        return {'y': data['y'].reshape(-1,128,128).transpose(-1,-2), 'sigmas': data['sigmas']}

    def __len__(self):
        return self.size