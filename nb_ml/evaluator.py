import yaml
import numpy as np
from sklearn.metrics import confusion_matrix
from keras.utils import np_utils
from sklearn.metrics import classification_report, accuracy_score
class evaluator():

    def __init__(self, pathToEvalConfigFile):
        self.config = {}
        self.__load_config(pathToEvalConfigFile)
        self.predictions = None
        self.first_predictions = None
        self.correct_labels = None
        self.confusion_matrix = None
        self.recall_class = None
        self.recall_tot = None
        self.precision_class = None
        self.precision_tot = None
        self.f1_class = None
        self.f1_tot = None
        self.accuracy = None

    def __load_config(self, pathToEvalConfigFile):
        with open(pathToEvalConfigFile, "r") as file:
            self.config = yaml.load(file)

    def get_predictions(self, predictions, correct_labels):

        prediction_lists2 = []
        for i in range(0, len(predictions[0])):
            prediction_lists2.append([])
        for prediction_list in predictions:
            for j in range(len(prediction_list)):
                prediction_lists2[j].append(prediction_list[j])
        prediction_lists = prediction_lists2

        self.predictions = prediction_lists
        self.first_predictions = prediction_lists[0]
        self.correct_labels = correct_labels


    def evaluate_prediction(self):
        num_classes = list(set(self.correct_labels))

        print(classification_report(self.correct_labels, self.first_predictions, num_classes))
        print("accuracy:" +str(accuracy_score(self.correct_labels,self.first_predictions)))
        print("Number of classes: "+str(num_classes))


    # def compute_precision(self):
    #     true_positives_vector = np.diag(self.confusion_matrix)
    #     with np.errstate(divide='ignore', invalid='ignore'):
    #         self.precision_class = true_positives_vector / np.sum(self.confusion_matrix, axis=0, dtype=np.float)
    #     self.precision_tot = np.sum(self.precision_class)/2
    # def compute_recall(self):
    #     a = np.diag(self.confusion_matrix)
    #     b = np.sum(self.confusion_matrix, axis=1, dtype=np.float)
    #     with np.errstate(divide='ignore', invalid='ignore'):
    #         self.recall = np.true_divide(a, b)
    #     self.recall_class = np.nan_to_num(self.recall)
    #     self.recall_tot = np.sum(self.recall_class)
    # def compute_f1(self):
    #     with np.errstate(divide='ignore', invalid='ignore'):
    #         self.f1_class = 2 * (self.precision_class * self.recall) / (self.precision_class + self.recall)
    #     self.f1_class = np.nan_to_num(self.f1_class)
    #     self.f1_tot = np.sum(self.f1_class)
    #
    # def compute_accuracy(self, true_positives, true_negatives, false_positives, false_negatives):
    #     self.accuracy = (true_positives+true_negatives)/(true_positives + true_negatives + false_positives + false_negatives)
    # def majority_rule(self):
    #     print("Something will come here")
    # def printKeyMetrics(self):
    #     print("Total precision: " + str(self.precision_tot))
    #     print("Total f1: " + str(self.f1_tot))
    #     print("Accuracy: "+ str(self.accuracy))
# if __name__ == '__main__':
#     evalTest = evaluator('/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/evaluator.yml')
#     correct_list = [1,1,1,1,1,1]
#     prediction_list = [1,0,1,0,1,0]
#     evalTest.evaluate_predictions(correct_list,prediction_list)
#
#     print("Fasit: " + str(correct_list))
#     print("Predicted list: " + str(prediction_list))
#
#     print("Accuracy:" + str(evalTest.accuracy))
#
#     print("Classwise recall: " + str(evalTest.recall_class))
#     print("Total recall: "+ str(evalTest.recall_tot))
#
#     print("Classwise precision: " + str(evalTest.precision_class))
#     print("Total precision: " + str(evalTest.precision_tot))
#
#
#     print("Classwise f1: " + str(evalTest.f1_class))
#     print("Total f1: " + str(evalTest.f1_tot))

