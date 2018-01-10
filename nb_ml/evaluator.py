import yaml
import numpy as np
from sklearn.metrics import confusion_matrix

class evaluator():

    def __init__(self, pathToConfigFile):
        self.load_config(pathToConfigFile = pathToConfigFile)
        self.confusion_matrix = None
        self.recall_class = None
        self.recall_tot = None
        self.precision_class = None
        self.precision_tot = None
        self.f1_class = None
        self.f1_tot = None
        self.accuracy = None
    def load_config(self, pathToConfigFile):
        with open(pathToConfigFile, "r") as file:
            self.config = yaml.load(file)

    def evaluate_predictions(self, correct_list, prediction_list):
        self.confusion_matrix = confusion_matrix(correct_list, prediction_list)
        self.compute_recall()
        self.compute_precision()

        true_positives_vector = []
        false_positives_vector = []
        false_negatives_vector = []
        true_negatives_vector = []
        for i in range(0, len(self.confusion_matrix)):
            TP_of_class_i = self.confusion_matrix[i, i]
            FP_of_class_i = np.sum(self.confusion_matrix, axis=0)[i] - self.confusion_matrix[i, i]  # The corresponding column for class_i - TP
            FN_of_class_i = np.sum(self.confusion_matrix, axis=1)[i] - self.confusion_matrix[i, i]  # The corresponding row for class_i - TP
            TN_of_class_i = np.sum(self.confusion_matrix) - TP_of_class_i - FP_of_class_i - FN_of_class_i
            true_positives_vector.append(TP_of_class_i)
            false_positives_vector.append(FP_of_class_i)
            false_negatives_vector.append(FN_of_class_i)
            true_negatives_vector.append(TN_of_class_i)

        true_positives = np.sum(true_positives_vector)
        false_positives = np.sum(false_positives_vector)
        false_negatives = np.sum(false_negatives_vector)
        true_negatives = np.sum(true_negatives_vector)
        self.compute_f1()
        self.compute_accuracy(true_positives,true_negatives,false_positives,false_negatives)

    def compute_precision(self):
        true_positives_vector = np.diag(self.confusion_matrix)
        self.precision_class = true_positives_vector / np.sum(self.confusion_matrix, axis=0, dtype=np.float)
        self.precision_tot = np.sum(self.precision_class)/2
    def compute_recall(self):
        a = np.diag(self.confusion_matrix)
        b = np.sum(self.confusion_matrix, axis=1, dtype=np.float)
        with np.errstate(divide='ignore', invalid='ignore'):
            self.recall = np.true_divide(a, b)
        self.recall_class = np.nan_to_num(self.recall)
        self.recall_tot = np.sum(self.recall_class)
    def compute_f1(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            self.f1_class = 2 * (self.precision_class * self.recall) / (self.precision_class + self.recall)
        self.f1_class = np.nan_to_num(self.f1_class)
        self.f1_tot = np.sum(self.f1_class)

    def compute_accuracy(self, true_positives, true_negatives, false_positives, false_negatives):
        self.accuracy = (true_positives+true_negatives)/(true_positives + true_negatives + false_positives + false_negatives)

if __name__ == '__main__':
    evalTest = evaluator('/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/evaluator.yml')
    correct_list = [1,1,1,1,1,1]
    prediction_list = [1,0,1,0,1,0]
    evalTest.evaluate_predictions(correct_list,prediction_list)

    print("Fasit: " + str(correct_list))
    print("Predicted list: " + str(prediction_list))

    print("Accuracy:" + str(evalTest.accuracy))

    print("Classwise recall: " + str(evalTest.recall_class))
    print("Total recall: "+ str(evalTest.recall_tot))

    print("Classwise precision: " + str(evalTest.precision_class))
    print("Total precision: " + str(evalTest.precision_tot))


    print("Classwise f1: " + str(evalTest.f1_class))
    print("Total f1: " + str(evalTest.f1_tot))

