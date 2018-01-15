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
        self.classification_report = None

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
        self.classification_report = classification_report(self.correct_labels, self.first_predictions, num_classes)
        self.accuracy = accuracy_score(self.correct_labels,self.first_predictions)

        print(self.classification_report)
        print("accuracy:" +str(self.accuracy))
        #print("accuracy:" +str(accuracy_score(self.correct_labels,self.first_predictions)))
        print("Number of classes: "+str(num_classes))

    def resultToLog(self, filepath):
        with open(filepath, 'w') as logFile:
            logFile.write(str(self.classification_report) + '\n\n' + "Accuracy: " + str(self.accuracy))


