from Step_1_CreateDataset.CreateDataset import *
import argparse
from pathlib import Path
import os



def main():
    if FLAGS.dataset == 'muse2':
        DATASET_PATH = Path('./Dataset/MUSE2/0. Initial Dataset')
        RESULT_PATH = Path('./Dataset/MUSE2/1. Aggregated')
        RESULT_PATH.mkdir(parents=True, exist_ok=True)

        GRANULARITY = 100

        for instance in os.scandir(DATASET_PATH):
            instance_path = instance.path
            print(f'Creating numerical datasets for {instance_path} using granularity {GRANULARITY}.')
            dataset = CreateDatasetMuse2(instance_path, GRANULARITY)
            dataset = dataset.add_data(instance_path, value_cols = ['Delta_TP9','Delta_AF7','Delta_AF8','Delta_TP10',
                                                                    'Theta_TP9','Theta_AF7','Theta_AF8','Theta_TP10',
                                                                    'Alpha_TP9','Alpha_AF7','Alpha_AF8','Alpha_TP10',
                                                                    'Beta_TP9','Beta_AF7','Beta_AF8','Beta_TP10',
                                                                    'Gamma_TP9','Gamma_AF7','Gamma_AF8','Gamma_TP10'],
                                    index_name='TimeStamp', label_name='label', aggregation='avg')

            dataset.to_csv(Path(str(RESULT_PATH) + '/' + instance.name))

    elif FLAGS.dataset == 'uci':
        DATASET_PATH = Path('./Dataset/UCI/0. Initial Dataset')
        RESULT_PATH = Path('./Dataset/UCI/1. Aggregated')
        RESULT_PATH.mkdir(parents=True, exist_ok=True)

        for instance in os.scandir(DATASET_PATH):
            instance_path = instance.path
            print(f'Creating numerical datasets for {instance_path}')
            dataset = CreateDatasetUCI(instance_path, value_cols = ['FP1','FP2','F7','F8','AF1','AF2','FZ','F4','F3','FC6','FC5','FC2','FC1','T8','T7','CZ',
                                                                    'C3','C4','CP5','CP6','CP1','CP2','P3','P4','PZ','P8','P7','PO2','PO1','O2','O1','X',
                                                                    'AF7','AF8','F5','F6','FT7','FT8','FPZ','FC4','FC3','C6','C5','F2','F1','TP8','TP7','AFZ',
                                                                    'CP3','CP4','P5','P6','C1','C2','PO7','PO8','FCZ','POZ','OZ','P2','P1','CPZ','nd','Y'])
            dataset = dataset.add_data()

            dataset.to_csv(Path(str(RESULT_PATH) + '/' + instance.name + '.csv'))

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help="Select which dataset to choose (default = Muse 2)", default='uci', choices=['muse2', 'uci'])

    FLAGS, unparsed = parser.parse_known_args()
    main()