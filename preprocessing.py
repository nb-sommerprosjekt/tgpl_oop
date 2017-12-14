#import preprocess_text_class
from shutil import copyfile
import math
from math import floor
from random import shuffle
from nltk.stem import snowball
from nltk.tokenize import word_tokenize
from numpy import array_split
import os
import yaml
#from preprocess_text_class import PreProcessRawText
#from data_augmentation import dataAugmentation
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import snowball
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import random
from math import floor
from numpy import array_split, array_str
from time import sleep
import time
import gensim
import random
import os
import pickle
from distutils.dir_util import copy_tree
import yaml
import logging
# class config(object):
#     def __init__(self):
#         self.config={}
#     def load_config(self, pathToConfigFile):
#         print("Loading config file: {}".format(pathToConfigFile))
#         #config={}
#         #self.load_config(pathToConfigFile)
#         with open(pathToConfigFile,"r") as file:
#             self.config = yaml.load(file)

class BaseData():
    #config = {}

    def __init__(self):
        self.config={}
        self.name_corpus = None
        self.data_set_name = None
        self.data_set_folder = None
        self.corpus_name_folder = None
        self.deweyDigitLength = None
        self.wiki_corpus_folder = None
        self.training_folder = None
        self.test_folder = None
        self.article_length = None
        self.split_folder = None
    def load_config(self, pathToConfigFile):

        #config={}
        #self.load_config(pathToConfigFile)
        with open(pathToConfigFile,"r") as file:
             self.config = yaml.load(file)
        self.name_corpus = self.config["name_corpus"]
        self.data_set_name = self.config["data_set_name"]
        self.data_set_folder = self.config["data_set_folder"]
        self.corpus_name_folder = os.path.join(self.data_set_folder, self.name_corpus)
        self.deweyDigitLength = self.config["deweyLength"]
        self.wiki_data_set_name = self.config["wiki_data_set_name"]
        #self.wiki_corpus_folder = os.path.join(self.corpus_name_folder, self.wiki_data_set_name)
        self.wiki_name_corpus = self.config["wiki_name_corpus"]
        self.training_folder = os.path.join(str(self.corpus_name_folder),str(self.name_corpus)+"_training")
        self.test_folder= os.path.join(self.corpus_name_folder,self.name_corpus+"_test")
        self.article_length = self.config["article_length"]
        self.split_folder = self.training_folder+"_split"
    def preprocess(self):


        rootdir=self.data_set_name
        #self.corpus_name_folder=os.path.join(self.data_set_folder, self.corpus_name)
        corpus_location=os.path.join(self.corpus_name_folder, self.name_corpus)
        if not os.path.exists(self.corpus_name_folder):
            os.makedirs(self.corpus_name_folder)
        if not os.path.exists(corpus_location):
            os.makedirs(corpus_location)
            print("Preprocessing the corpus from the folder '{}'.".format(rootdir))

            #self.text_processing(rootdir, corpus_location)

            self.text_processing(rootdir, corpus_location)
            print("Preprocessed corpus saved in the folder {}".format(corpus_location))



    def preprocess_wiki(self):

        rootdir=self.wiki_data_set_name
        #self.corpus_name_folder=os.path.join(self.data_set_folder, self.corpus_name)

        corpus_location=os.path.join(self.corpus_name_folder, self.wiki_name_corpus)
        if not os.path.exists(self.corpus_name_folder):
            os.makedirs(self.corpus_name_folder)
        if not os.path.exists(corpus_location):
            os.makedirs(corpus_location)
            print("Preprocessing the corpus from the folder '{}'.".format(rootdir))

            #self.text_processing(rootdir, corpus_location)

            self.text_processing_wiki(rootdir, corpus_location)
            print("Preprocessed corpus saved in the folder {}".format(corpus_location))

    def text_processing(self,rootdir, corpus_location):
        counter = 0
        print("Starter lesing av filer")
        for subdir, dirs, files in os.walk(rootdir):
            #print(files)
            for file in files:
                #print(file)
                if str(file)[:5] == "meta-":
                    counter += 1
                    if counter % 100 == 0:
                        print("File {} out of {}".format(counter, len(files) / 2))
                    f = open(os.path.join(rootdir, file), "r+")
                    for line in f.readlines():
                        if "dewey:::" in line:
                            dewey = line.split(":::")[1]
                            dewey = dewey.strip()
                            #print(dewey)
                    f = open(os.path.join(rootdir, file[5:]), "r")
                    text = f.read()

                    processed_text = PreProcessRawText(text)
                    if self.config["sentences"]:
                        processed_text.remove_punctuation()
                    if self.config["stop_words"]:
                        processed_text.remove_stopwords()
                    if self.config["stemming"]:
                        # norStem = snowball.NorwegianStemmer()
                        processed_text.stem_text()
                    if self.config["lower_case"]:
                        processed_text.processed_text_lower()
                    if self.config["extra_functions"]:
                        # add function calls here
                        # FUNCTION_CALL()
                        # FUNCTION_CALL()
                        # FUNCTION_CALL()
                        pass
                    f.close()
                    file = open(os.path.join(corpus_location, file[5:]), "w")
                    file.write("__label__" + dewey + " " + processed_text.processed_text)

    def text_processing_wiki(self,rootdir, corpus_location):
        counter = 0


        for subdir, dirs, files in os.walk(rootdir):
            for file in files:
                counter += 1
                if counter % 100 == 0:
                    print("File {} out of {}".format(counter, len(files)))
                # print(subdir)
                f = open(os.path.join(rootdir, file), "r+")
                text = f.read()

                text = text.split(" ")
                dewey = text[0].replace("__label__", "")
                text = " ".join(text[1:])
                processed_text = PreProcessRawText(text)
                if self.config["sentences"]:
                    processed_text.remove_punctuation()
                if self.config["stop_words"]:
                    processed_text.remove_stopwords()
                if self.config["stemming"]:
                    # norStem = snowball.NorwegianStemmer()
                    processed_text.stem_text()
                if self.config["lower_case"]:
                    processed_text.processed_text_lower()
                if self.config["extra_functions"]:
                    # add function calls here
                    # FUNCTION_CALL()
                    # FUNCTION_CALL()
                    # FUNCTION_CALL()
                    pass
                f2 = open(os.path.join(corpus_location, file), "w+")
                f2.write("__label__" + dewey + " " + text)
        print("Preprocessed  wiki_corpus saved in the folder {}".format(self.wiki_name_corpus))
        #return wiki_corpus_folder  # Kanskje kjøre neste steg.
        #return wiki_corpus_folder  # Kanskje kjøre neste steg.


    def split_to_training_and_test(self):
        print("Splitting to training and test:")
        training_folder= os.path.join(str(self.corpus_name_folder),str(self.name_corpus)+"_training")
        test_folder= os.path.join(self.corpus_name_folder,self.name_corpus+"_test")
        corpus_name_location=os.path.join(self.corpus_name_folder ,self.name_corpus)

        dewey_dict={}

        for subdir, dirs, files in os.walk(corpus_name_location):
            for file in files:
                f = open(os.path.join(corpus_name_location, file), "r+")
                text = f.read()

                dewey = text.split(" ")[0].replace("__label__", "")
                dewey=dewey.replace(".","")
                if self.deweyDigitLength>0:
                    if len(dewey)>self.deweyDigitLength:
                        dewey=dewey[:self.deweyDigitLength]

                if dewey in dewey_dict:
                    dewey_dict[dewey].append(file)
                else:
                    dewey_dict[dewey]=[file]
        training_list=[]
        test_list= []

        for key in dewey_dict.keys():
            temp_list=dewey_dict[key]
            shuffle(temp_list)
            split = max(1, math.floor(len(temp_list) * self.config["test_ratio"]))
            if len(temp_list)>1:
                test_list.extend(temp_list[:split])
                training_list.extend(temp_list[split:])
            else:
                training_list.extend(temp_list)


        if  os.path.exists(training_folder):
            print("The test-folder already exists. No action will be taken. ")
            print("Training set is  {} articles.".format(len(training_list)))
        else:
            os.makedirs(training_folder)
            print("Training set is  {} articles.".format(len(training_list)))
            for file in training_list:
                copyfile(os.path.join(corpus_name_location, file), os.path.join(training_folder, file))

        if  os.path.exists(test_folder):
            print("The test-folder already exists. No action will be taken. ")
            print("Test set is  {} articles.".format(len(test_list)))

        else:
            os.makedirs(test_folder)
            print("Test set is  {} articles.".format(len(test_list)))
            for file in test_list:
                copyfile(os.path.join(corpus_name_location, file), os.path.join(test_folder, file))


        print("Splitting: Complete.")
        self.training_folder = training_folder
        self.test_folder = test_folder

    def add_wiki_to_training(self):
        print("Adding wiki data to training set. ")
        for subdir, dirs, files in os.walk(self.wiki_name_corpus):
            for file in files:
                print(file)
                copyfile(os.path.join(self.wiki_name_corpus, file), os.path.join(self.training_folder, file))
        for subdir,dirs,files in os.walk(self.training_folder):
            print("New total in training set: {}".format(len(files) ))
            break
        print("Wiki data: Complete.")



    def split_articles(self):
        folder = self.training_folder
        self.article_length=int(self.article_length)
        #folder_split=folder+"_split"

        if  os.path.exists(self.split_folder):
            print("The split-folder already exists. No action will be taken. ")
            #return folder_split
        else:
            os.makedirs(self.split_folder)

        for subdir, dirs, files in os.walk(folder):
            for file in files:
                dewey, text = self.read_dewey_and_text(os.path.join(folder, file))
                texts=self.split_text(text,self.article_length)
                for i,text in enumerate(texts):

                    temp = list(text)
                    text=" ".join(temp)
                    #print(i, len(text))
                    #print(os.path.join(folder_split,file[:-4]+"_"+str(i)+file[-4:]))
                    with open(os.path.join(self.split_folder,file[:-4]+"_"+str(i)+file[-4:]),"w+") as f:
                        f.write("__label__"+dewey+" "+str(text))
        self.training_folder = self.split_folder

    def read_dewey_and_text(self, location):
        with  open(location, "r+")as f:
            text = f.read()
        text = text.split(" ")
        dewey = text[0].replace("__label__", "")
        text = " ".join(text[1:])
        return dewey, text

    def split_text(self, text, number_of_words_per_output_article):
        tokenized_text = word_tokenize(text, language="norwegian")
        split_count = max(1, floor(len(tokenized_text)/int(number_of_words_per_output_article)))
        split_texts = array_split(tokenized_text,split_count)
        texts=[]
        for t in split_texts:
            t=list(t)
            t=" ".join(t)
            texts.append(t)
        return texts




class PreProcessRawText:
    processed_text = None
    def __init__(self, raw_text):
        global processed_text
        self.processed_text = raw_text

    def remove_punctuation(self):
        self.processed_text = re.sub('[^a-zA-ZæøåÆØÅ]+', ' ', self.processed_text)
        self.processed_text = self.processed_text.replace("  ", " ")


    def remove_stopwords(self):
        tokens = word_tokenize(self.processed_text)
        filtered_words = [word for word in tokens if word not in set(stopwords.words('norwegian'))]

        self.processed_text = ' '.join(filtered_words)

    def stem_text(self):
        tokens = word_tokenize(self.processed_text)
        norStem = snowball.NorwegianStemmer()

        stemmed_words = list()
        for word in tokens:
            stemmed_words.append(norStem.stem(word))

        self.processed_text = ' '.join(stemmed_words)

    def processed_text_lower(self):
        self.processed_text.lower()



class dataAugmention:

    def __init__(self):
        self.config = {}
        self.noise_percentage = 10
        self.da_splits = 10
        self.MostSimilarPickle = None
        self.noise_method = "uniform_noise"

    def getConfig(self, pathToConfigFile):
        with open(pathToConfigFile,"r") as file:
             self.config = yaml.load(file)
        self.noise_percentage = self.config["da_noise_percentage"]
        self.da_splits = self.config["da_splits"]
        self.noise_method = self.config["da_noise_method"]
        with open(self.config["PathToNoisePickle"], 'rb') as handle:
            self.MostSimilarPickle = pickle.load(handle)


if __name__ == '__main__':
    logging.basicConfig(filename='run.log', level=logging.INFO)
    logging.info('Started')


    train_test = BaseData()
    train_test.load_config("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/preprocess.yml")
    print(train_test.config)

    train_test.preprocess()
    train_test.split_to_training_and_test()

    #Add wikipedia data to corpus
    train_test.preprocess_wiki()
    train_test.add_wiki_to_training()

    # Split articles|
    train_test.split_articles()
    logging.info('Finished')
