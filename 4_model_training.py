import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from Step_4_MLmodels.PrepareDatasetForLearning import PrepareDatasetForLearning
from Step_4_MLmodels.LearningAlgorithms import ClassificationAlgorithms
from Step_4_MLmodels.Evaluation import ClassificationEvaluation
from Step_4_MLmodels.FeatureSelection import FeatureSelectionClassification
from util import util
from util.VisualizeDataset import VisualizeDataset

# Set up file names and locations.
FOLDER_PATH = Path('./Dataset/3. Features Extracted')
RESULT_PATH = Path('./Dataset/4. Saved Models')

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

def main():
    print_flags()

    # We'll create an instance of our visualization class to plot results.
    DataViz = VisualizeDataset(__file__)

    RESULT_PATH.mkdir(exist_ok=True, parents=True)
    # for this script, we want to first load in all datasets
    # since the Prepare dataset function accepts a list of pd dataframes
    prepare = PrepareDatasetForLearning()
    all_datasets = []

    dataset = pd.read_csv('./Dataset/3. Features Extracted/drunk1.csv', index_col=0)
    dataset.index = pd.to_datetime(dataset.index)
    
    train_X, train_y, test_X, test_y, val_X, val_y = prepare.split_normal_dataset(FOLDER_PATH, val_size=0)
    train_X = np.array(train_X)
    train_y = np.array(train_y)
    test_X = np.array(test_X)
    test_y = np.array(test_y)
    val_X = np.array(val_X)
    val_y = np.array(val_y)

    selected_features = ['Delta_AF7_temp_max_ws_10', 'Alpha_TP9_temp_mean_ws_10', 'Delta_AF7_temp_slope_ws_30', 'FastICA_2', 
    'Alpha_TP9_temp_median_ws_20', 'Delta_AF8_temp_max_ws_10', 'Beta_TP10_freq_30.0_Hz_ws_10', 
    'Beta_TP9_temp_std_ws_20', 'Theta_TP10_temp_max_ws_20', 'Gamma_TP9_temp_median_ws_20',
     'Gamma_TP10_freq_30.0_Hz_ws_10', 'Alpha_TP10_temp_std_ws_20', 'Gamma_AF7_freq_30.0_Hz_ws_10', 'Delta_TP10', 
     'Beta_TP9_temp_median_ws_20', 'Delta_TP10_temp_min_ws_20', 'Theta_TP9_temp_median_ws_30', 
     'Delta_AF8_temp_min_ws_20', 'Delta_AF8_temp_mean_ws_10', 'Beta_TP9_freq_0.0_Hz_ws_10']

    # select subsets of features which we will consider:
    pca_features = ['pca_1','pca_2','pca_3','pca_4']
    ica_features = ['FastICA_1','FastICA_2','FastICA_3','FastICA_4','FastICA_5','FastICA_6','FastICA_7','FastICA_8','FastICA_9','FastICA_10',
    'FastICA_11','FastICA_12','FastICA_13','FastICA_14','FastICA_15','FastICA_16','FastICA_17','FastICA_18','FastICA_19','FastICA_20']
    time_features = [name for name in dataset.columns if '_temp_' in name]
    freq_features = [name for name in dataset.columns if (('_freq' in name) or ('_pse' in name))]

    # feature selection below we will use as input for our models:
    basic_features = ['Delta_TP9','Delta_AF7','Delta_AF8','Delta_TP10','Theta_TP9','Theta_AF7','Theta_AF8','Theta_TP10','Alpha_TP9','Alpha_AF7',
    'Alpha_AF8','Alpha_TP10','Beta_TP9','Beta_AF7','Beta_AF8','Beta_TP10','Gamma_TP9','Gamma_AF7','Gamma_AF8','Gamma_TP10']
    basic_w_PCA = list(set().union(basic_features, pca_features))
    basic_w_ICA = list(set().union(basic_features, ica_features))
    all_features = list(set().union(basic_features, ica_features, time_features, freq_features))

    num_features = 20

    possible_feature_sets = [basic_features, basic_w_PCA, basic_w_ICA, all_features, selected_features]
    feature_names = ['initial set', 'basic_w_PCA', 'basic_w_ICA', 'all_features', 'Selected features']

    learner = ClassificationAlgorithms()
    eval = ClassificationEvaluation()

    if FLAGS.mode == 'rf':
        pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y = learner.random_forest(train_X, train_y, test_X, gridsearch=False, save_model=False)
        eval.confusion_matrix(pred_test_y, test_y, [0,1], FLAGS.mode)
        eval.print_all(pred_test_y, test_y)

    elif FLAGS.mode == 'lstm':
        pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y = learner.lstm(train_X, train_y, test_X, save_model=False)
        pred_test_y = np.ravel(pred_test_y)
        depth, base_length = learner.find_depth_and_base(np.shape(train_X)[0], np.shape(test_X)[0])
        test_y = test_y[:base_length]
        test_y = test_y.reshape(1, base_length, 1)
        test_y = np.ravel(test_y)
        eval.confusion_matrix(pred_test_y, test_y, [0,1], FLAGS.mode)
        eval.print_all(pred_test_y, test_y)


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help="Select which model to train", choices=['lstm', 'rf'])

    FLAGS, unparsed = parser.parse_known_args()
    main()