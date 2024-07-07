import torch

from torch.utils.data import Dataset

import os

import pandas as pd

class weather_dataset(Dataset):
    def __init__(self, seq_len=300, pred_len=100, base="/weather/", set=None, times=None):
        self.seq_len = seq_len
        self.pred_len = pred_len
        if set is not None:
            self.data = set
            self.times = times
            return
        current_dir = os.getcwd()
        files = os.listdir(current_dir + base)
        csv_files = [file for file in files if file.endswith('.csv')]
        print("Found files:", csv_files)
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

        pred = torch.tensor(self.data[i+self.seq_len:i+self.seq_len+self.pred_len])
        pred_times = torch.tensor(self.times[i+self.seq_len:i+self.seq_len+self.pred_len].astype(int))

        return seq_times.float(), sequence.float(), pred_times.float(), pred.float()
        
    def split(self, train_size=0.8): # split the current instance in train, test and validation sets where the train set is 80% of all the data and the remaining 20% is equally divided in test and validation. In this case since we have a big quantity of data, it is enough to have a test and validation set of only 10%
        split_index = int(len(self.data)*train_size)
        test_size = int((self.__len__() - split_index)/2)
        train_set = weather_dataset(seq_len=self.seq_len, pred_len=self.pred_len, set=self.data[:split_index], times=self.times[:split_index])
        test_set = weather_dataset(seq_len=self.seq_len, pred_len=self.pred_len, set=self.data[split_index:split_index+test_size], times=self.times[split_index:split_index+test_size])
        valid_set = weather_dataset(seq_len=self.seq_len, pred_len=self.pred_len, set=self.data[split_index+test_size:], times=self.times[split_index+test_size:])

        return train_set, test_set, valid_set
    
    def parse_time(self, df): # parse the time that is initially stored in the format 'dd.MM.yyyy hh:mm:ss' to a pandas dataframe that contains ['month','day','weekday','hour','minute'] in separate columns. The paper uses the same parsing strategy
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