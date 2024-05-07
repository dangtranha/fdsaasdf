from Create_dataset import create_dataset
from ClassificationAlgorithms import *
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def main(dataset_mode = "default"):
    X_train, X_test, y_train, y_test = create_dataset(option=dataset_mode)
    class_test_y = lstm(X_train, y_train, X_test, y_test)
    confusionmatrix_rf_final = confusion_matrix(y_test, class_test_y, labels=[0,1])
    cm_display = ConfusionMatrixDisplay(confusion_matrix = confusionmatrix_rf_final, display_labels = ["non-drunk", "drunk"])

    cm_display.plot(cmap="Blues")
    plt.show()
    cm_display.figure_.savefig('lstm_confusion_matrix.png')
    print("Accuracy:", accuracy_score(y_test, class_test_y))
    print("Precision:" , precision_score(y_test, class_test_y))
    print("Recall: ", recall_score(y_test, class_test_y))
    print("F1 Score: ", f1_score(y_true=y_test, y_pred=class_test_y))
        
if __name__ == "__main__":
    main(dataset_mode="default")