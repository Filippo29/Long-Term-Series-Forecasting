import torch
import torch.nn as nn

import numpy as np

from torch.utils.data import Dataset

import os
import csv

from datetime import datetime, timedelta

class weather_dataset(Dataset):
    def __init__(self, seq_len=100, pred_len=200, base="/weather/", device=None, set=None):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.device = device
        if set is not None:
            self.data = set
            return
        current_dir = os.getcwd()
        files = os.listdir(current_dir + base)
        csv_files = [file for file in files if file.endswith('.csv')]
        print("Found files:", csv_files)
        self.data = np.empty((0, 22), float)
        base_date = datetime(1970, 1, 1)
        delta = timedelta(milliseconds=1)
        for csv_file in csv_files:
            with open(current_dir + base + csv_file, 'r', encoding='ISO-8859-1') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)
                for row in csv_reader:
                    utc_time = datetime.strptime(row[0], "%d.%m.%Y %H:%M:%S")
                    row[0] = (utc_time - base_date) // delta
                    for i in range(1, len(row)):
                        row[i] = float(row[i])
                    self.data = np.vstack([self.data, row])
        assert len(self.data) > seq_len + pred_len
        print("Found a total of {} samples.".format(len(self.data)))
    
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len

    def __getitem__(self, i):
        sequence = self.data[i:i+self.seq_len]
        seq_times = torch.tensor(sequence[:, 0].astype(int))
        sequence = torch.tensor(sequence[:, 1:])

        pred = self.data[i+self.seq_len:i+self.seq_len+self.pred_len]
        pred_times = torch.tensor(pred[:, 0].astype(int))
        pred = torch.tensor(pred[:, 1:])

        if self.device is not None:
            seq_times.to(self.device)
            sequence.to(self.device)
            pred_times.to(self.device)
            pred.to(self.device)
        return seq_times, sequence, pred_times, pred
        
    def split(self, train_size=0.8):
        split_index = int(len(self.data)*train_size)
        test_size = int((self.__len__() - split_index)/2)
        train_set = weather_dataset(set=self.data[:split_index], device=self.device)
        test_set = weather_dataset(set=self.data[split_index:split_index+test_size], device=self.device)
        valid_set = weather_dataset(set=self.data[split_index+test_size:], device=self.device)

        return train_set, test_set, valid_set