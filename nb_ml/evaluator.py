import yaml
import numpy as np
from sklearn.metrics import confusion_matrix

class evaluator():

    def __init__(self, pathToConfigFile):
        self.load_config(pathToConfigFile = pathToConfigFile)
        self.recall = None
        self.precision = None
        self.f1 = None
        self.accuracy = None
    def load_config(self, pathToConfigFile):
        with open(pathToConfigFile, "r") as file:
            self.config = yaml.load(file)

    def evaluate_predictions(self, correct_list, prediction_list):
        c = confusion_matrix(correct_list, prediction_list)
        true_positives_vector = np.diag(c)
        self.precision = true_positives_vector / np.sum(c, axis=0, dtype=np.float)
        self.recall = np.diag(c) / np.sum(c, axis=1, dtype=np.float)
        # print(precision_vector)
        # print(recall_vector)

        true_positives_vector = []
        false_positives_vector = []
        false_negatives_vector = []
        true_negatives_vector = []
        for i in range(0, len(c)):
            TP_of_class_i = c[i, i]
            FP_of_class_i = np.sum(c, axis=0)[i] - c[i, i]  # The corresponding column for class_i - TP
            FN_of_class_i = np.sum(c, axis=1)[i] - c[i, i]  # The corresponding row for class_i - TP
            TN_of_class_i = np.sum(c) - TP_of_class_i - FP_of_class_i - FN_of_class_i
            true_positives_vector.append(TP_of_class_i)
            false_positives_vector.append(FP_of_class_i)
            false_negatives_vector.append(FN_of_class_i)
            true_negatives_vector.append(TN_of_class_i)

        print (true_positives_vector)
        print(false_negatives_vector)
        print(false_positives_vector)
        print(true_negatives_vector)

    def compute_f1(self):
        self.f1 = 2 * (self.precision * self.recall) / (self.precision + self.recall)
    def compute_accuracy(self, true_positives, true_negatives, false_positives, false_negatives):
        self.accuracy = (true_positives+true_negatives)/(true_positives + true_negatives + false_positives + false_negatives)
if __name__ == '__main__':
    evalTest = evaluator('/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/evaluator.yml')
