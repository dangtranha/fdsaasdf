from sklearn import metrics
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import inspect

# Class for evaluation metrics of classification problems.
class ClassificationEvaluation:

    # Returns the accuracy given the true and predicted values.
    def accuracy(self, y_true, y_pred):
        return metrics.accuracy_score(y_true, y_pred)

    # Returns the precision given the true and predicted values.
    # Note that it returns the precision per class.
    def precision(self, y_true, y_pred):
        return metrics.precision_score(y_true, y_pred, average=None)

    # Returns the recall given the true and predicted values.
    # Note that it returns the recall per class.
    def recall(self, y_true, y_pred):
        return metrics.recall_score(y_true, y_pred, average=None)

    # Returns the f1 given the true and predicted values.
    # Note that it returns the recall per class.
    def f1(self, y_true, y_pred):
        return metrics.f1_score(y_true, y_pred, average='macro')

    # Returns the area under the curve given the true and predicted values.
    # Note: we expect a binary classification problem here(!)
    def auc(self, y_true, y_pred_prob):
        return metrics.roc_auc_score(y_true, y_pred_prob)

    # Returns the confusion matrix given the true and predicted values.
    def confusion_matrix(self, y_true, y_pred, labels, alg_name, save_figure=False):
        confusionmatrix = metrics.confusion_matrix(y_true, y_pred, labels=labels)
        if save_figure:
            cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusionmatrix, display_labels = ["non-drunk", "drunk"])
            cm_display.plot(cmap="Blues")
            plt.show()
            cm_display.figure_.savefig(str(alg_name) + 'confusion_matrix.png')

    def print_all(self, y_true, y_pred):
        print("Accuracy:", metrics.accuracy_score(y_true, y_pred))
        print("Precision:" , metrics.precision_score(y_true, y_pred))
        print("Recall: ", metrics.recall_score(y_true, y_pred))
        print("F1 Score: ", metrics.f1_score(y_true=y_true, y_pred=y_pred))