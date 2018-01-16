import yaml
import utils
from evaluator import evaluator
import fasttext
import datetime
import os
class fast_text(evaluator):
    def __init__(self, pathToConfigFile):
        print("Something will be written here")
        self.__config = {}
        self.load_config(pathToConfigFile)

        self.x_train = None
        self.y_train = None
        self.fasttext_train_input = ""
        self.model = None

        super(fast_text, self).__init__(self.evaluatorConfigPath)
    def load_config(self, pathToConfigFile):
        with open(pathToConfigFile,"r") as file:
             self.__config = yaml.load(file)
        self.minNumArticlesPerDewey = self.__config["minNumArticlesPerDewey"]
        self.trainingSetPath = self.__config["trainingSetPath"]
        self.evaluatorConfigPath = self.__config["evaluatorConfigPath"]
        self.epochs = self.__config["epochs"]
        self.learningRate = self.__config["learningRate"]
        self.lrUpdate = self.__config["lrUpdate"]
        self.lossFunction = self.__config["lossFunction"]
        self.wikiVec = self.__config["wikiVec"]
        self.wikiPath = self.__config["wikiPath"]
        self.wordWindow = self.__config["wordWindow"]
        self.kLabels = self.__config["kLabels"]
        self.modelsDirectory = self.__config["modelsDirectory"]
        timestamp = '{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())
        self.save_path = self.modelsDirectory + "/fasttext-" + timestamp
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.tmp_ft_file_path =self.save_path + "/tmp.txt"
        self.tmp_ft_test_file_path =  self.save_path + "tmp_test.txt"

    def fit(self):
        print("Henter inn tekst")
        self.trainFolder2fasttext()
        print("Starter trening")
        self.model = fasttext.supervised(self.tmp_ft_file_path, 'model')
        os.remove(self.tmp_ft_file_path)
    def predict(self, pathToTestSet):
        self.testFolder2Fasttext(pathToTestSet)
        result = (self.model).test(self.tmp_ft_test_file_path)
        os.remove(self.tmp_ft_test_file_path)

        precision = result.precision
        recall = result.recall
        print("Accuracy: {}".format(precision))

        print("Recall: {}".format(recall))
        print("smth different")

    def trainFolder2fasttext(self):
        corpus_df = utils.get_articles_from_folder(self.trainingSetPath)
        ###Filtering articles by frequency of articles per dewey
        corpus_df = corpus_df.groupby('dewey')['text', 'file_name', 'dewey'].filter(
                    lambda x: len(x) >= self.minNumArticlesPerDewey)
        self.y_train = corpus_df["dewey"].values
        self.x_train = corpus_df["text"].values
        self.findValidDeweysFT()
        fasttextInputFile = open(self.tmp_ft_file_path, "w")
        for i in range(0,len(self.y_train)):
            fasttextInputFile.write("__label__"+str(self.y_train[i])+" " + str(self.x_train[i]) + '\n')
        fasttextInputFile.close()
    def testFolder2Fasttext(self, pathToTestSet):
        test_corpus_df = utils.get_articles_from_folder(pathToTestSet)
        ###Filtering articles by frequency of articles per dewey
        test_corpus_df = test_corpus_df.loc[test_corpus_df['dewey'].isin(self.validDeweys)]
        self.y_test = test_corpus_df['dewey'].values
        self.x_test = test_corpus_df['text'].values

        fasttextInputFile = open(self.tmp_ft_test_file_path, "w")
        for i in range(0,len(self.y_test)):
            fasttextInputFile.write("__label__"+str(self.y_test[i])+" " + str(self.x_test[i]) + '\n')
        fasttextInputFile.close()
    def findValidDeweysFT(self):
        self.validDeweys = list(set(self.y_train))

if __name__ == '__main__':
    test = fast_text("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/fasttext.yml")
    test.fit()
    test.predict("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/data_set/test_fredag_mlp/test_fredag_mlp_test")
    print(test.validDeweys)
    #test.folder2fasttext()