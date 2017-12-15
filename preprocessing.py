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
import sys
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
        self.pathToInputFolder = None
        self.text_array = []
        self.dewey_array = []
        self.text_names = []
        self.output_folder = None

    def getConfig(self, pathToConfigFile, pathToInputFolder):
        with open(pathToConfigFile,"r") as file:
             self.config = yaml.load(file)
        self.noise_percentage = self.config["da_noise_percentage"]
        self.da_splits = self.config["da_splits"]
        self.noise_method = self.config["da_noise_method"]
        self.pathToInputFolder = pathToInputFolder
        self.output_folder = self.config["data_set_name"]+"/"+self.config["name_corpus"]+"/"+self.config["corpus_name_folder"]+"/"+self.config["artificial_folder"]
        with open(self.config["PathToNoisePickle"], 'rb') as handle:
            self.MostSimilarPickle = pickle.load(handle)
        #self.text_array = pathToInputFolder


    def get_articles(self):
    #     # Tar inn en folder som er labelet på fasttext-format. Gir ut to arrays. Et med deweys og et med tekstene. [deweys],[texts]
        #arr = os.listdir(folder)
        folder = self.pathToInputFolder
        arr_txt = [path for path in os.listdir(folder) if path.endswith(".txt")]
        arr_txt.sort()

        for article_path in arr_txt:
                article = open(folder+'/'+article_path, "r")
                article = article.readlines()
                for article_content in article:
                    dewey=article_content.partition(' ')[0].replace("__label__","")
                    text  = article_content.replace("__label__"+dewey,"")
                    self.text_array.append(text)
                    self.dewey_array.append(dewey[:3])
        self.text_names = [path.replace('.txt','') for path in arr_txt ]

    def create_fake_corpus(self):
        if self.da_splits==0:
            print("Antallet splits må være høyere enn 0, sjekk config")
            sys.exit()
        print("Tekstdata på plass")
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        text_list = list(enumerate(self.text_names))
        total_number_of_texts = len(self.text_names)
        for index, text_name in text_list:
            tokenized_text = word_tokenize(self.text_array[index],language= 'norwegian')

            # print("teksten er tokenized")
            # print("Legger til støy og lager nye tekster")

            if self.noise_method == "noise_on_parts":
                full_texts_with_noise = self.add_synonyms_to_parts_of_text(tokenized_text,self.da_splits, self.noise_percentage)
            elif self.noise_method =="uniform_noise":
                full_texts_with_noise = self.add_synonyms_randomly_to_text(tokenized_text, self.da_splits, self.noise_percentage)
            else:
                print("Du har valgt en støymetode som ikke finnes. Programmet avsluttes")
                break


            # print("Støy er lagt til")

            # print("Starter utskrift av tekster")

            for noise_text_index, noise_text in enumerate(full_texts_with_noise):

                noise_text_w_label = "__label__" + str(self.dewey_array[index]) + ' ' + noise_text
                noise_split_file = open(self.output_folder + '/' + text_name.replace('_split','') + '_' + str(noise_text_index)+".txt",'w')
                noise_split_file.write(noise_text_w_label)
                noise_split_file.close()
            if (index+1%1000==0):
                print("Tekst nr " + str(index+1) + "/" + str(total_number_of_texts)+ " er ferdig prosessert")
            #Copying original texts into folder with artificial ones.



    def add_synonyms_to_parts_of_text(self,tokenized_text,number_of_splits,percentage):
        '''Module for adding noise to text. To add noise to the whole text, set number_of_splits=1. Number of documents output is set by number_of_splits.
        If number_of_splits=10, 10 documents will be returned where different parts of the documents have been induced with a percentage of noise set by percentage.'''
        new_percentage = percentage*10
        start_split_arrays = time.time()
        array_of_text_parts = array_split(tokenized_text, self.da_splits)
        end_split_arrays = time.time()
        array_of_text_parts_with_synonyms = []
        print("Split_Arrays_time:" + str(end_split_arrays -start_split_arrays))

        # Making noise array
        start_making_noise_array = time.time()

        total_words_replaced = 0
        for part_text in array_of_text_parts:
            synonym_text, words_replaced= self.replace_words_with_synonyms(part_text, new_percentage)
            total_words_replaced +=words_replaced
            array_of_text_parts_with_synonyms.append(synonym_text)

        end_making_noise_arrays = time.time()
        print("Making noise arrays: " + str( end_making_noise_arrays - start_making_noise_array))
        print(str(total_words_replaced)+ " har blitt erstattet av " + str(len(tokenized_text)*10))

         # Making new full tekst with the noise induced fractions included
        full_texts_with_noise_induced_on_parts_array = []

       # print(array_of_text_parts_with_synonyms)
        for i in range(0,len(array_of_text_parts_with_synonyms)):

            temp = list(array_of_text_parts)

            temp[i] = array_of_text_parts_with_synonyms[i]
            flat_temp = []
            for sublist in temp:
                for item in sublist:
                     flat_temp.append(item)

            temp_text = ' '.join(flat_temp)

            full_texts_with_noise_induced_on_parts_array.append(temp_text)

        return full_texts_with_noise_induced_on_parts_array

    def add_synonyms_randomly_to_text(self,tokenized_text, num_splits, noise_percentage):
        noise_texts_array = []

        for i in range(0, num_splits-1):
            noise_texts_array.append(self.add_synonyms_to_parts_of_text(tokenized_text,1,noise_percentage)[0])
        return noise_texts_array

    def replace_words_with_synonyms(self,tokenized_text,percentage):
        ''' Takes tokenized text as input, replaces words with synonyms or word2vec and outputs tokenized text as string with synonyms replaced'''
        num_words_to_replace = floor((percentage / 100) * len(tokenized_text))
        synonym_text = []

        words_replaced = 0
        synonyms_put_into_text = []
        # Replacing words with word from norsk synonymordbok
        for word in tokenized_text:
            synonym = self.find_synonyms(word.lower())
            if len(synonym) > 0 and words_replaced<(num_words_to_replace):
                synonym_text.append(synonym[0])
                #synonyms_put_into_text.append("!!!støy-->"+word+"-->" + synonym[0]+"<--støy!!!")
                synonyms_put_into_text.append(synonym[0])
                words_replaced = words_replaced + 1
                #print("Synonymordbok:"+ word + "---> " + str(synonym[0]))
            else:
                synonym_text.append(word)
        # Replacing remaining words with words from word2vec
        if (words_replaced < (num_words_to_replace)):
            words_left_to_replace = (num_words_to_replace-words_replaced)
            random_elements = random.sample(range(0,len(synonym_text)-1), len(synonym_text)-1)
            for index in random_elements:
                words_left_to_replace = (num_words_to_replace - words_replaced)
                if synonym_text[index] not in synonyms_put_into_text and words_left_to_replace>0:
                    synonym_w2v = self.get_similar_words_from_word2vec(synonym_text[index], self.MostSimilarPickle)
                    if len(synonym_w2v)>0:
                        temp_show = synonym_text[index]
                        # synonym_text[index] = "!!!støy--> "+temp_show+"-->"+synonym_w2v +"<--Støy!!!"
                        synonym_text[index] = synonym_w2v
                        words_replaced=words_replaced+1
        return synonym_text , words_replaced

    def find_synonyms(self, word):

        '''Takes a string as input and returns a array of synonyms in norwegian, if no synonyms exists, a empty array will be returned.'''
        class MyWordNet:
            def __init__(self, wn):
                self._wordnet = wn

            def synsets(self, word, pos=None, lang="nob"):
                return self._wordnet.synsets(word, pos=pos, lang = lang)

        wn = MyWordNet(wordnet)
        synonyms = []
        synonym_collection =[]
        for syn in wn.synsets(word, lang="nob"):
            for l in syn.lemmas(lang="nob"):
                #synonyms.append(l.name())
                if l.name() != word: #l.name() not in synonyms and :
                    synonyms.append(l.name())
                    synonym_collection.append(l.name())


        # if len(synonyms) == 0:
        #     #print("bruker nynorsk")
        #     ##Checking if synonyms exists in NNO if there was none in NOB.
        #     wnn = MyWordNet(wordnet)
        #     for syn in wnn.synsets(word, lang="nno"):
        #         for l in syn.lemmas(lang="nno"):
        #             if   l.name() != word: #and l.name() not in synonyms:
        #                 synonyms.append(l.name())
        #                 synonym_collection.append(l.name())

        #print(word + "---> " + str(synonym_collection))
        return synonyms

    def get_similar_words_from_word2vec(self,word, mostSimilarPickle):
        '''Function taking a word and a word2vec model as input, outputting the most similar word.'''
        try:
            most_similar_word = mostSimilarPickle[word]
        except KeyError:
            most_similar_word = []
        # print(word + "---> " + str(most_similar_word))
        return most_similar_word



    def copyArtificialFolderIntoCorpus(self):
         copy_tree(self.output_folder, self.pathToInputFolder)
    #
    # def make_vocab(array_of_texts):
    #     vocab = set()
    #     total_words = 0
    #     for text in array_of_texts:
    #         tokenize_text = text.split(" ")
    #
    #         total_words+=len(tokenize_text)
    #         for word in tokenize_text:
    #                     vocab.add(word.lower())
    #     print(vocab)
    #     print(len(vocab))
    #     print(str(total_words))
    #     return vocab
    # def make_most_similar_dictionary(vocab_set, w2v_model):
    #     # function which takes vocab as input, queries the w2v model for each word in vocab and gets the most similar
    #     # word semantically and saves it in a dictionary.
    #     most_similar = {}
    #     progress_count = 0
    #     vocab_len = len(vocab_set)
    #     array = list(vocab_set)
    #     for word in array:
    #         progress_count +=1
    #         try:
    #             similar_words = w2v_model.wv.similar_by_word(word.lower(), topn=1)
    #             most_similar_word = similar_words[0][0]
    #         except KeyError:
    #             most_similar_word = []
    #             print(word + "finnes ikke i vokabularet")
    #
    #         if len(most_similar_word) >0:
    #             most_similar[word] = most_similar_word
    #         print(str(progress_count)+"/" +str(vocab_len) + " er fullført" )
    #     print(len(most_similar))
    #
    #     with open('most_similar_dict.pickle', 'wb') as handle:
    #         pickle.dump(most_similar, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #     print(len(most_similar))
    #     return most_similar


    # def remove_words_from_text(tokenized_text, percentage):
    #     ''' Chooses random words in the tokenized text and removes them. The amount is chosen by the percentage provided. It is not recommended to remove more than 10 percent. Returns reduced tokenized string.'''
    #     num_elements_to_remove = floor((percentage/100) * len(tokenized_text))
    #     words_to_remove = random.sample(tokenized_text, num_elements_to_remove)
    #
    #     reduced_text = tokenized_text
    #     for words in words_to_remove:
    #         reduced_text.remove(words)
    #     return ' '.join(reduced_text)
    #
    # def remove_parts_of_text(tokenized_text, percentage):
    #     '''Removes a random part of the input-tokenized text, size set by the percentage, and outputs a reduced tokenized text.'''
    #     number_of_splits = floor(100/percentage)
    #     array_of_text_parts = array_split(tokenized_text, number_of_splits)
    #     del array_of_text_parts[random.randint(0, len(array_of_text_parts) - 1)]
    #     reduced_array = [item for sublist in array_of_text_parts for item in sublist]
    #
    #     return reduced_array
    #
    # def add_words_to_text(tokenized_text, percentage, which_dewey_is_this_from):
    #     ''' Adds words randomly to text. Words are chosen either from word2vec or tfidf-matrix relevant to the articles dewey. This function returns the modified tokenized_text'''
    #




if __name__ == '__main__':
    logging.basicConfig(filename='run.log', level=logging.INFO)
    logging.info('Started')


    # train_test = BaseData()
    # train_test.load_config("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/preprocess.yml")
    # print(train_test.config)
    #
    # train_test.preprocess()
    # train_test.split_to_training_and_test()
    #
    # #Add wikipedia data to corpus
    # train_test.preprocess_wiki()
    # train_test.add_wiki_to_training()
    #
    # # Split articles|
    # train_test.split_articles()

    test = dataAugmention()
    test.getConfig("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/preprocess.yml", "/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/data_set/test_torsdag_mlp/test_torsdag_mlp_training_split" )
    test.get_articles()
    test.create_fake_corpus()
    test.copyArtificialFolderIntoCorpus()
    #print(test.output_folder)
    #test.get_articles()
    #print(len(test.text_array))
    #print(len(test.text_names))
    #print(len(test.dewey_array))
    logging.info('Finished')
