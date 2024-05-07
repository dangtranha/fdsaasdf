import numpy as np
import pandas as pd
import os
from datetime import timedelta
from scipy import stats

class CreateDatasetMuse2:
    root_dir = ''
    granularity = 0

    def __init__(self, root_dir, granularity):
        self.root_dir = root_dir
        self.granularity = granularity

    def create_empty_dataset(self, start, end, cols):
        timestamps = pd.date_range(start, end, freq=str(self.granularity)+'ms')
        dataframe = pd.DataFrame(index=timestamps, columns=cols)
        for col in cols:
            dataframe[str(col)] = np.nan
        return dataframe
    
    def add_num_value(self, dataset, dataframe, value_cols, index_name, aggregation='avg'):
        for i in range(0, len(dataframe.index)):
            rows = dataset[
                (dataset[index_name] >= dataframe.index[i]) &
                (dataset[index_name] < (dataframe.index[i] + timedelta(milliseconds=self.granularity)))]

            for col in value_cols:
                if len(rows) > 0:
                    dataframe.loc[dataframe.index[i], str(col)] = np.average(rows[col])
                else:
                    dataframe.loc[dataframe.index[i], str(col)] = np.nan
                    
        return dataframe                    

    def add_label(self, dataset, dataframe, value_cols, index_name, label_name):
        for i in range(0, len(dataframe.index)):

            rows = dataset[
                (dataset[index_name] >= dataframe.index[i]) &
                (dataset[index_name] < dataframe.index[i] + timedelta(milliseconds=self.granularity))]

            for col in value_cols:
                if len(rows) > 0:
                    dataframe.loc[dataframe.index[i], str(col)] = stats.mode(rows[col])[0]
                else:
                    dataframe.loc[dataframe.index[i], str(col)] = np.nan
        return dataframe

    def add_data(self, file, value_cols, index_name, label_name, aggregation='avg'):
        dataset = pd.read_csv(file, skipinitialspace=True)
        dataset[index_name] = pd.to_datetime(dataset[index_name])
        dataset.dropna(thresh=dataset.shape[1]-10,axis=0, inplace=True)

        dataframe = self.create_empty_dataset(min(dataset[index_name]), max(dataset[index_name]), value_cols)
        dataframe = self.add_num_value(dataset, dataframe, value_cols, index_name, aggregation)
        dataframe = self.add_label(dataset, dataframe, value_cols, index_name, label_name)

        return dataframe
        
class CreateDatasetUCI:
    instance_path = ''
    value_cols = []
    channels = {}

    def __init__(self, instance_path, value_cols):
        self.instance_path = instance_path
        self.value_cols = value_cols
        self.channels = {channel_name : [] for channel_name in self.value_cols}

    def add_num_value(self):
        for file in os.scandir(self.instance_path):
            file_path = file.path
            with open(file_path, 'r') as file:
                lines = file.readlines()[4:]
            for line in lines:
                splitted = line.split()
                if line.startswith('#'):
                    channel_name = splitted[1]
                else:
                    self.channels[channel_name].append(float(splitted[-1]))

    def add_label(self, dataset):
        dirs = self.instance_path.split()
        folder_name = dirs[-1] 
        label_type = folder_name[3]
        if label_type == 'a':           # Drunk
            dataset['label'] = 1
        elif label_type == 'c':         # Non-drunk
            dataset['label'] = 0
        else:
            raise Exception("File name label error, not 'c' or 'a'")
        return dataset

    def add_data(self):
        self.add_num_value()
        df = pd.DataFrame(self.channels)
        df = self.add_label(df)
        return df



# def create_dataset(option = "default", WINDOW_SIZE = 30, train_ratio = 0.8, DRUNK_PATH = DRUNK_PATH, NONDRUNK_PATH = NONDRUNK_PATH, DRUNK_FILE_PATH = DRUNK_FILE_PATH, NONDRUNK_FILE_PATH = NONDRUNK_FILE_PATH):
#     if option == "interval":
#         all_drunk_dataset = []
#         all_nondrunk_dataset = []

#         for instance in os.scandir(DRUNK_PATH): 
#             instance_path = instance.path
#             dataset = pd.read_csv(instance_path, index_col=0)
#             dataset.index = pd.to_datetime(dataset.index)
#             dataset = dataset.drop(columns='check')
#             all_drunk_dataset.append(dataset)

#         for instance in os.scandir(NONDRUNK_PATH): 
#             instance_path = instance.path
#             dataset = pd.read_csv(instance_path, index_col=0)
#             dataset.index = pd.to_datetime(dataset.index)
#             dataset = dataset.drop(columns='check')
#             all_nondrunk_dataset.append(dataset)

#         X = []
#         y = []

#         for i in range (len(all_drunk_dataset)):
#             df = all_drunk_dataset[i].to_numpy()
#             print(len(df))
#             for j in range (0,len(df) - WINDOW_SIZE):
#                 col = [a for a in df[j:j+WINDOW_SIZE]]
#                 X.append(col)
#                 y.append(1)

#         for i in range (len(all_nondrunk_dataset)):
#             df = all_nondrunk_dataset[i].to_numpy()
#             print(len(df))
#             for j in range (0,len(df) - WINDOW_SIZE):
#                 col = [a for a in df[j:j+WINDOW_SIZE]]
#                 X.append(col)
#                 y.append(0)
            
#         X_df = pd.DataFrame(data=X)
#         y_df = pd.DataFrame(data=y)
#         X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=1-train_ratio)
#         # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=(1-train_ratio)/train_ratio)
#     else:
#         drunk_df = pd.read_csv(DRUNK_FILE_PATH)
#         nondrunk_df = pd.read_csv(NONDRUNK_FILE_PATH)

#         drunk_df = drunk_df.dropna(axis=0)
#         nondrunk_df = nondrunk_df.dropna(axis=0)

#         y_drunk = drunk_df['check']
#         X_drunk = drunk_df.drop(columns='check')
#         X_drunk = X_drunk.drop(columns=X_drunk.columns[0])

#         y_nondrunk = nondrunk_df['check']
#         X_nondrunk = nondrunk_df.drop(columns='check')
#         X_nondrunk = X_nondrunk.drop(columns=X_nondrunk.columns[0])

#         X_drunk, y_drunk = X_drunk[:2500], y_drunk[:2500]
#         X_nondrunk, y_nondrunk = X_nondrunk[:2500], y_nondrunk[:2500]

#         X_train_drunk, X_test_drunk, y_train_drunk, y_test_drunk = train_test_split(X_drunk, y_drunk, test_size=0.2)
#         X_train_nondrunk, X_test_nondrunk, y_train_nondrunk, y_test_nondrunk = train_test_split(X_nondrunk, y_nondrunk, test_size=0.2)
        
#         X_train = pd.concat([X_train_drunk, X_train_nondrunk])
#         X_test = pd.concat([X_test_drunk, X_test_nondrunk])
#         y_train = pd.concat([y_train_drunk, y_train_nondrunk])
#         y_test = pd.concat([y_test_drunk, y_test_nondrunk])

#     return X_train, X_test, y_train, y_test