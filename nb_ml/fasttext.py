import yaml
import utils
from evaluator import evaluator
class fasttext(evaluator):
    def __init__(self, pathToConfigFile):
        print("Something will be written here")
        self.load_config(pathToConfigFile)
        self.x_train = None
        self.y_train = None
        self.fasttext_train_input = ""
        super(fasttext, self).__init__(self.evaluatorConfigPath)
    def load_config(self, pathToConfigFile):
        with open(pathToConfigFile,"r") as file:
             self.__config = yaml.load(file)
        self.minNumArticlesPerDewey = self.__config["minNumArticlesPerDewey"]
        self.trainingSetPath = self.__config["trainingSetPath"]
        self.evaluatorConfigPath = self.__config["evaluatorConfigPath"]

    def fit(self):
        print("smth else")

    def predict(self):
        print("smth different")

    def folder2fasttext(self):
        corpus_df = utils.get_articles_from_folder(self.trainingSetPath)
        ###Filtering articles by frequency of articles per dewey
        corpus_df = corpus_df.groupby('dewey')['text', 'file_name', 'dewey'].filter(
                    lambda x: len(x) >= self.minNumArticlesPerDewey)
        self.y_train = corpus_df["dewey"].values
        self.x_train = corpus_df["text"].values
        print(len(self.y_train))
        print(len(self.x_train))
        #for i in range(0,len(self.x_train)):
        for i in range(0,len(self.y_train)):
            self.fasttext_train_input +="__label__"+str(self.y_train)+" " + str(self.x_train) + '\n'
        print(self.fasttext_train_input)
        print(len(self.fasttext_train_input.split('\n')))
if __name__ == '__main__':
    test = fasttext("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/fasttext.yml")
    test.folder2fasttext()