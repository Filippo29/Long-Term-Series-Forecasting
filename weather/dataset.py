import torch
import torch.nn as nn

from torch.utils.data import Dataset

import os
import csv

import random

from datetime import datetime, timedelta

class weather_dataset(Dataset):
    def __init__(self, base="/weather/", device=None, set=None):
        self.device = device
        if set is not None:
            self.data = set
            return
        current_dir = os.getcwd()
        files = os.listdir(current_dir + base)
        csv_files = [file for file in files if file.endswith('.csv')]
        print("Found files:", csv_files)
        self.data = []
        base_date = datetime(1970, 1, 1)
        delta = timedelta(milliseconds=1)
        for csv_file in csv_files:
            with open(current_dir + base + csv_file, 'r', encoding='ISO-8859-1') as file:
                csv_reader = csv.reader(file)
                next(csv_reader)
                for row in csv_reader:
                    self.data.append(row)
                    utc_time = datetime.strptime(self.data[-1][0], "%d.%m.%Y %H:%M:%S")
                    self.data[-1][0] = (utc_time - base_date) // delta
        print("Found a total of {} samples.".format(len(self.data)))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        sample = self.data[i]
        for i in range(1, len(sample)):
            sample[i] = float(sample[i])
        time = torch.tensor(sample[0])
        data = torch.tensor(sample[1:])
        if self.device is not None:
            time.to(self.device)
            data.to(self.device)
        return time, data # date in milliseconds, sample
        
    def split(self, train_size=0.8):
        tmp_data = [s[:] for s in self.data]
        random.shuffle(tmp_data)

        split_index = int(len(tmp_data)*train_size)
        train_set = weather_dataset(set=tmp_data[:split_index], device=self.device)
        test_set = weather_dataset(set=tmp_data[split_index:], device=self.device)

        return train_set, test_set