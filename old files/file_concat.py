import pandas as pd
from pathlib import Path
import os
NON_PATH = Path('Dataset/step3_result/non_drunk')
DRUNK_PATH = Path('Dataset/step3_result/drunk')
drunk_datasets = []
nondrunk_datasets = []
all_datasets = []

for instance in os.scandir(NON_PATH): 
    instance_path = instance.path
    dataset = pd.read_csv(instance_path, index_col=0)
    dataset.index = pd.to_datetime(dataset.index)
    nondrunk_datasets.append(dataset)
    all_datasets.append(dataset)

for instance in os.scandir(DRUNK_PATH):
    instance_path = instance.path
    dataset = pd.read_csv(instance_path, index_col=0)
    dataset.index = pd.to_datetime(dataset.index)
    drunk_datasets.append(dataset)
    all_datasets.append(dataset)

all = pd.concat(all_datasets, axis=0)
drunk = pd.concat(drunk_datasets, axis=0)
nondrunk = pd.concat(nondrunk_datasets, axis=0)

all.to_csv('Dataset/step3_result/concatenated/all.csv')
drunk.to_csv('Dataset/step3_result/concatenated/drunk.csv')
nondrunk.to_csv('Dataset/step3_result/concatenated/nondrunk.csv')