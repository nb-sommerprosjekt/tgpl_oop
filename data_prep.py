import preprocess_text_class
from shutil import copyfile
import math
from math import floor
from random import shuffle
from nltk.stem import snowball
from nltk.tokenize import word_tokenize
from numpy import array_split
import os
import yaml
from preprocess_text_class import PreProcessRawText
#from data_augmentation import dataAugmentation

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
                            print(dewey)
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

#if __name__ == '__main__':


    # # # Initialize corpus
    # train_test = BaseData()
    # train_test.load_config("config/preprocess.yml")
    # print(train_test.config)

    # train_test.preprocess()
    # train_test.split_to_training_and_test()
    #
    # #Add wikipedia data to corpus
    # train_test.preprocess_wiki()
    # train_test.add_wiki_to_training()
    #
    # # Split articles
    # train_test.split_articles()
    #

# def read_dewey_and_text(location):
#     with  open(location, "r+")as f:
#         text = f.read()
#     text = text.split(" ")
#     dewey = text[0].replace("__label__", "")
#     text = " ".join(text[1:])
#     return dewey,text


# def remove_unecessary_articles(article_folder, corpus_folder, minimum_articles, dewey_digits):
#     dewey_digits=int(dewey_digits)
#     rubbish_folder=os.path.join(corpus_folder,"rubbish")
#     if  os.path.exists(rubbish_folder):
#         # shutil.rmtree(training_folder_split)
#         # os.makedirs(training_folder_split)
#         print("The rubbish-folder already exists. No action will be taken. ")
#     else:
#         print("Creating rubbish-folder")
#         os.makedirs(rubbish_folder)
#
#     dewey_dict = {}
#
#
#     for subdir, dirs, files in os.walk(article_folder):
#         for file in files:
#             dewey, text = read_dewey_and_text(os.path.join(article_folder, file))
#             if int(dewey_digits) > 0:
#                 if len(dewey) > int(dewey_digits):
#                     dewey = dewey[:dewey_digits]
#
#             if dewey in dewey_dict:
#                 dewey_dict[dewey].append(file)
#             else:
#                 dewey_dict[dewey] = [file]
#     valid_deweys=set()
#     set_length=0
#     for key in dewey_dict.keys():
#         if len(dewey_dict[key])<int(minimum_articles):
#             for file in dewey_dict[key]:
#                 os.rename(os.path.join(article_folder, file), os.path.join(rubbish_folder, file))
#         else:
#             for file in dewey_dict[key]:
#                 valid_deweys.add(key)
#                 set_length+=1
#                 dewey,text=read_dewey_and_text(os.path.join(article_folder, file))
#                 if len(dewey) > dewey_digits:
#                     dewey = dewey[:dewey_digits]
#                 with open(os.path.join(article_folder, file), "r+") as f3:
#                     f3.seek(0)
#                     f3.write("__label__" + dewey + " " + str(text))
#                     f3.truncate()
#                     f3.close()
#
#     print("Removed unnecessary dewey numbers. There are {} unique dewey numbers in the training set.".format(len(valid_deweys)))
#     return rubbish_folder,valid_deweys,set_length
#
#
# def prep_test_set(test_folder,valid_deweys,article_length,dewey_digits, rubbish_folder):
#     print("Prepping test-folder.")
#     dewey_digits=int(dewey_digits)
#     #test_folder=split_articles(test_folder,article_length)
#
#     print("Test-folder split.")
#     test_files_split=[]
#     test_set_length=0
#     for subdir, dirs, files in os.walk(test_folder):
#         for file in files:
#             #print(len(files))
#             with open(os.path.join(test_folder, file), "r+") as f:
#                 text = f.read()
#                 text=text.split(" ")
#                 dewey = text[0].replace("__label__", "")
#                 text=" ".join(text[1:])
#                 dewey = dewey.replace(".", "")
#                 if len(dewey) > int(dewey_digits):
#                     dewey = dewey[:dewey_digits]
#                     #print(dewey)
#                 #print(dewey)
#                 if dewey in valid_deweys:
#                     test_files_split.append([dewey,list(split_text(text,article_length))])
#                     test_set_length+=1
#                     #print("Not thrash")
#                     f.seek(0)
#                     f.write("__label__" + dewey + " " + str(text))
#                     f.truncate()
#                 else:
#                     os.rename(os.path.join(test_folder, file), os.path.join(rubbish_folder, file))
#     return test_folder,test_set_length,test_files_split
#
#
#
#
#
# def load_set(folder):
#     print("Loading {} now".format(folder))
#     total_tekst = ""
#     counter = 0
#
#     for subdir, dirs, files in os.walk(folder):
#         for file in files:
#
#             if counter % 1000 == 0:
#                 print("Done {} out of {}".format(counter, len(files)))
#             counter += 1
#             with open(os.path.join(folder, file), "r+") as f:
#                 f.seek(0)
#                 text = f.read()
#             total_tekst +=  text + '\n'
#     return total_tekst
#
# def create_folder(path):
#     os.makedirs(path, exist_ok=True)
#
# def save_file(location,name,text):
#     with open (os.path.join(location,name),"w+") as file:
#         file.write(text)
#     return  str(os.path.join(location,name))

