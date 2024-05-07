from sklearn.model_selection import train_test_split
import numpy as np
import random
import copy
import pandas as pd
import os
from pathlib import Path

class PrepareDatasetForLearning:
    class_col = 'label'
    
    def split_normal_dataset(self, FOLDER_PATH, test_size=0.2, val_size=0.1):
        """_summary_
        Args:
            FOLDER_PATH (_type_): Path to the datasets
            test_size (float, optional): Desired test size. Defaults to 0.2.
            val_size (float, optional): Desired evaluation size. Defaults to 0.1.
        Returns:
            _type_: X_train, y_train, X_test, y_test, X_val, y_val
        """
        X_train = []
        X_test = []
        X_val = []
        y_train = []
        y_test = []
        y_val = []


        all_datasets = []
        label = []
        for instance in os.scandir(FOLDER_PATH): # go through all instances of experiments
            instance_path = instance.path
            dataset = pd.read_csv(instance_path, index_col=0)
            dataset.index = pd.to_datetime(dataset.index)
            temp = []
            if "non" in instance.name:
                for i in range(len(dataset)):
                    label.append(0)
            else:  
                for i in range(len(dataset)):
                    label.append(1)
            all_datasets.append(dataset)

        all_datasets = pd.concat(all_datasets)

        X_train, X_test, y_train, y_test = train_test_split(all_datasets, label, test_size=test_size)
        if val_size != 0:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size/(1-test_size))
        
        print('Successfully splitting the dataset, here is the information:')
        print('X_train shape: ', np.shape(X_train))
        print('y_train shape: ', np.shape(y_train))
        print('X_test shape: ', np.shape(X_test))
        print('y_test shape: ', np.shape(y_test))
        print('X_val shape: ', np.shape(X_val))
        print('y_val shape: ', np.shape(y_val))
        print('----------------------------------------------------------')

        return X_train, y_train, X_test, y_test, X_val, y_val