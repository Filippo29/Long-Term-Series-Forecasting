import torch
import torch.nn as nn

import numpy as np

from torch.utils.data import Dataset

import os

import pandas as pd

class weather_dataset(Dataset):
    def __init__(self, seq_len=100, pred_len=200, base="/weather/", device=None, set=None, times=None):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.device = device
        if set is not None:
            self.data = set
            self.times = times
            return
        current_dir = os.getcwd()
        files = os.listdir(current_dir + base)
        csv_files = [file for file in files if file.endswith('.csv')]
        print("Found files:", csv_files)
        self.data = np.empty((0, 22), float)
        self.data = None
        self.times = None
        for csv_file in csv_files:
            self.data = pd.concat([self.data, pd.read_csv(current_dir + base + csv_file, encoding = "ISO-8859-1")])
        self.times = self.parse_time(self.data)
        date_col_name = 'Date Time'

        self.data = self.data.drop(columns=[date_col_name])
        self.data = self.data.values.astype(float)
        self.times = self.times.values

        assert len(self.data) > seq_len + pred_len and len(self.times) == len(self.data)
        print("Found a total of {} samples.".format(len(self.data)))
    
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len

    def __getitem__(self, i):
        sequence = self.data[i:i+self.seq_len]
        seq_times = torch.tensor(self.times[i:i+self.seq_len].astype(int))
        sequence = torch.tensor(sequence)

        pred = self.data[i+self.seq_len:i+self.seq_len+self.pred_len]
        pred_times = torch.tensor(self.times[i+self.seq_len:i+self.seq_len+self.pred_len].astype(int))

        if self.device is not None:
            seq_times.to(self.device)
            sequence.to(self.device)
            pred_times.to(self.device)
            pred.to(self.device)
        return seq_times, sequence, pred_times, pred
        
    def split(self, train_size=0.8):
        split_index = int(len(self.data)*train_size)
        test_size = int((self.__len__() - split_index)/2)
        train_set = weather_dataset(set=self.data[:split_index], times=self.times[:split_index], device=self.device)
        test_set = weather_dataset(set=self.data[split_index:split_index+test_size], times=self.times[split_index:split_index+test_size], device=self.device)
        valid_set = weather_dataset(set=self.data[split_index+test_size:], times=self.times[split_index+test_size:], device=self.device)

        return train_set, test_set, valid_set
    
    def parse_time(self, df):
        dates_df = df.copy()
        dates_df["Date Time"] = pd.to_datetime(dates_df["Date Time"], dayfirst=True)
        dates_df['month'] = dates_df["Date Time"].apply(lambda row:row.month,1)
        dates_df['day'] = dates_df["Date Time"].apply(lambda row:row.day,1)
        dates_df['weekday'] = dates_df["Date Time"].apply(lambda row:row.weekday(),1)
        dates_df['hour'] = dates_df["Date Time"].apply(lambda row:row.hour,1)
        dates_df['minute'] = dates_df["Date Time"].apply(lambda row:row.minute,1)
        dates_df['minute'] = dates_df.minute.map(lambda x:x//15)
        freq_map = ['month','day','weekday','hour','minute']
        return dates_df[freq_map]