import pandas as pd
import numpy as np
import glob
from sklearn.mixture import GaussianMixture
from datetime import timedelta

class Csv_process:
    def __init__(self):
        self.final_df = []
        self.temp_df = []
        self.final_nondrunk = []
        self.final_drunk = []
        self.csv_path = './Datasets/Processed'
        self.nondrunk_files = glob.glob(self.csv_path + './nondrunk/non*.csv')
        self.drunk_files = glob.glob(self.csv_path + './drunk/drunk*.csv')
    
    # Read in files, drop irrelevant columns, add label and save the concatenated file
    def concat_nondrunk(self):
        nondrunk_df = []
        for file in self.nondrunk_files:
            csv_file = pd.read_csv(file)
            # dropping = np.r_[24:38]
            # csv_file.drop(csv_file.columns[dropping], axis=1, inplace=True)
            # csv_file['label'] = 0
            nondrunk_df.append(csv_file )
            self.temp_df.append(csv_file)
        self.final_nondrunk = pd.concat(nondrunk_df, axis=0)
        self.final_nondrunk.to_csv('./Datasets/EEG_nondrunk.csv', index=False)

    def concat_drunk(self):
        drunk_df = []
        for file in self.drunk_files:
            csv_file = pd.read_csv(file)  
            # dropping = np.r_[24:38]
            # csv_file.drop(csv_file.columns[dropping], axis=1, inplace=True)
            # csv_file['label'] = 1                                      
            drunk_df.append(csv_file)
            self.temp_df.append(csv_file)
        self.final_drunk = pd.concat(drunk_df, axis=0, ignore_index=True)
        self.final_drunk.to_csv('./Datasets/EEG_drunk.csv', index=False)

    # Save nondrunk and drunk into the final dataset
    def concat_all(self):
        final_df = pd.concat(self.temp_df, axis=0)
        final_df.to_csv('./Datasets/EEG_dataset.csv', index=False)

class Dataset_process:
    def __init__(self, granularity):
        self.granularity = granularity
    
    def create_dataset(self, start_time, end_time, cols):
        timestamps = pd.date_range(start=start_time, end=end_time, freq=str(self.granularity)+'ms')
        df = pd.DataFrame(index=timestamps, columns=cols)
        for col in cols:
            df[str(col)] = np.nan
        return df
    
    def add_value(self, data_table, dataset, cols):
        for i in  range(0, len(data_table.index)):
            print(i)
            relevant_rows = dataset[(dataset['TimeStamp'] >= data_table.index[i]) &
                                    (dataset['TimeStamp'] < (data_table.index[i]
                                     + timedelta(milliseconds=self.granularity)))]  
            for col in cols:
                if len(relevant_rows) > 0:
                    data_table.loc[data_table.index[i], str(col)] = np.average(relevant_rows[col])
                else:
                    data_table.loc[data_table.index[i], str(col)] = np.nan 
        return data_table

    def add_data(self, cols):
        dataset = pd.read_csv('EEG_dataset.csv')
        print('read successfully')
        dataset['TimeStamp'] = pd.to_datetime(dataset['TimeStamp'])
        dataset.dropna(thresh=dataset.shape[1]-10,axis=0, inplace=True)
        print('drop successfully')
        data_table = self.create_dataset(min(dataset['TimeStamp']), max(dataset['TimeStamp']), cols)
        print('created dataset')
        data_table = self.add_value(data_table, dataset,cols)
        print('added value')
        return data_table
    
