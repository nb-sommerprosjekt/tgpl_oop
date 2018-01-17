import yaml
import utils
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import numpy as np
import datetime
import os
import dill
from evaluator import evaluator
class logReg(evaluator):

    def __init__(self, pathToConfigFile):
        self.config = {}
        self.load_config(pathToConfigFile = pathToConfigFile)
        self.x_train = []
        self.y_train = []
        self.correct_deweys = None
        self.validDeweys = []
        self.model = None
        self.predictions = None
        self.accuracy = None

    def load_config(self, pathToConfigFile):
        with open(pathToConfigFile, "r") as file:
            self.config = yaml.load(file)
        self.training_set = self.config["training_set"]
        self.test_set = self.config["test_set"]
        self.vectorizationType = self.config["vectorizationType"]
        self.minNumArticlesPerDewey = self.config["minNumArticlesPerDewey"]
        self.kPreds = self.config["kPreds"]
        self.modelsDirectory =self.config["modelsDirectory"]
        self.evaluatorConfigPath = self.config["evaluatorConfigPath"]
        super(logReg, self).__init__(self.evaluatorConfigPath)

    def fit_LogReg(self):
        self.fasttext2sklearn()
        #tfidf = TfidfVectorizer(norm = 'l2', min_df = 2, use_idf = True, smooth_idf= False, sublinear_tf = True, ngram_range = (1,4),
        #                        max_features = 20000)

        if self.vectorizationType == "tfidf":
            vectorizer = TfidfVectorizer()
            print("starter transformering")
            x_train_vectorized = vectorizer.fit_transform(self.x_train)
        else:
            if self.vectorizationType == "count":
                vectorizer = CountVectorizer()
                x_train_vectorized = vectorizer.fit_transform(self.x_train)
        print("Transformering gjennomført")
        test_corpus_df = utils.get_articles_from_folder(self.test_set)
        test_corpus_df = test_corpus_df.loc[test_corpus_df['dewey'].isin(self.validDeweys)]

        self.y_test = test_corpus_df['dewey']
        self.x_test = test_corpus_df['text']
        self.correct_deweys = test_corpus_df['dewey'].values

        x_test_vectorized = vectorizer.transform(self.x_test)
        self.x_test = x_test_vectorized
        print("Starter trening")

        mod = LogisticRegression()
        mod.fit(x_train_vectorized, self.y_train)
        #self.model = logMod
        self.model = mod
        self.saveModel()

    def predict(self):
        self.getPredictionsAndAccuracy()

    def printPredictionsAndAccuracy(self):
        print(self.predictions)
        print(self.accuracy)


    def saveModel(self):

        model_name = "model.pickle"
        timestamp = '{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())
        save_path = self.modelsDirectory + "/logReg-" + self.vectorizationType + timestamp

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        model_save_file = open(save_path + "/" + model_name, 'wb')
        dill.dump(self.model, model_save_file, -1)
        print("modell_lagret")
        model_path = save_path + "/model.pickle"
        print("Modellen er lagret i :"+ model_path)
    def getPredictionsAndAccuracy(self):


        predictions = []
        topNpredictions = []

        if self.x_test.shape[0] > 0 and self.y_test.shape[0] > 0 and self.model is not None:
            for text in self.x_test:

                predictions.append(self.model.predict(text))
                pred_proba = self.model.predict_proba(text)
                n = self.kPreds
                topN_prob_indexes = np.argsort(pred_proba)[:, :-n - 1:-1]
                for val in topN_prob_indexes:
                    print(self.model.classes_[val])
                    topNpredictions.append(self.model.classes_[val])


            accuracy = accuracy_score(self.y_test, predictions)
            print("preds fra sklearn:" + str(predictions))
        else:
            print("Input var ikke riktig. Sjekk om modell og  testsett eksisterer")

        self.accuracy = accuracy
        self.predictions =topNpredictions
        return predictions, accuracy, topNpredictions


    def fasttext2sklearn(self):

            corpus_df = utils.get_articles_from_folder(self.training_set)
            ###Filtering articles by frequency of articles per dewey
            corpus_df = corpus_df.groupby('dewey')['text', 'file_name', 'dewey'].filter(lambda x: len(x) >= self.minNumArticlesPerDewey)
            self.y_train = corpus_df['dewey']
            self.x_train = corpus_df['text']
            self.findValidDeweysSklearn()

    def findValidDeweysSklearn(self):
        self.validDeweys = list(set(self.y_train))

    def run_evaluation(self):
        super(logReg, self).get_predictions(self.predictions, self.correct_deweys)
        super(logReg, self).evaluate_prediction()

# if __name__ == '__main__':
#     print("starting")
#     test = logReg("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/logreg.yml")
#     test.fit_LogReg()
#     test.predict()
#     test.printPredictionsAndAccuracy()
    #self.predictions = []
    #self.accuracy = None
    #self.topNpredictions = []


