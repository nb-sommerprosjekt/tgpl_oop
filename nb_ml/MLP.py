# INSPIRERT av MLP fra http://nadbordrozd.github.io/blog/2017/08/12/looking-for-the-text-top-model/

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Dropout, Activation
import numpy as np
from keras.models import Sequential, load_model
import time
import os
import datetime
import pickle
import re
#import evaluator2
from sklearn.metrics import accuracy_score
import utils_nb
import yaml
import gc

from evaluator import evaluator

class mlp(evaluator):


    def __init__(self, pathToConfig):
        #vocab = []
        #vocab_size = 0
        self.config = {}
        self.load_config(pathToConfig)
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.correct_deweys = None
        self.predictions = []
        self.accuracy = None

        super(mlp, self).__init__(self.evaluatorConfigPath)
        #MAX_SEQUENCE_LENGTH, \
        #EPOCHS, \
        #FOLDER_TO_SAVE_MODEL, \
        #LOSS_MODEL,
        #VECTORIZATION_TYPE,
        #VALIDATION_SPLIT)
    def load_config(self, pathToConfigFile):


        #self.load_config(pathToConfigFile)
        with open(pathToConfigFile,"r") as file:
             self.__config = yaml.load(file)

        self.trainingSetPath=self.__config["trainingSetPath"]
        self.vocabSize = self.__config["vocabSize"]
        self.maxSequenceLength = self.__config["maxSequenceLength"]

        self.batchSize = self.__config["batchSize"]
        self.vectorizationType = self.__config["vectorizationType"]
        self.epochs = self.__config["epochs"]
        self.validationSplit = self.__config["validationSplit"]
        self.folderToSaveModels = self.__config["folderToSaveModels"]
        self.modelDir = None
        self.lossModel = self.__config["lossModel"]
        self.minNumArticlesPerDewey = self.__config["minNumArticlesPerDewey"]
        self.kPreds = self.__config["kPreds"]
        self.evaluatorConfigPath = self.__config["evaluatorConfigPath"]
        self.strictArticleSelection = self.__config["strictArticleSelection"]
    def fit(self):
        '''Training model'''

        start_time= time.time()
        print(self.trainingSetPath)
        self.x_train, self.y_train, tokenizer, num_classes, labels_index = self.fasttextTrain2mlp(self.trainingSetPath, self.maxSequenceLength,
                                                                                   self.vocabSize, self.vectorizationType,
                                                                                    minNumArticlesPerDewey = self.minNumArticlesPerDewey)

        model = Sequential()
        model.add(Dense(512, input_shape=(self.maxSequenceLength,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(512, input_shape=(self.vocabSize,)))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))

        model.summary()
        model.compile(loss=self.lossModel,
                      optimizer='adam',
                      metrics=['accuracy'])

        model_history = model.fit(self.x_train, self.y_train,
                  batch_size=self.batchSize,
                  epochs=self.epochs,
                  verbose=1,
                  validation_split=self.validationSplit
                  )
        #utils_nb.plotTrainHistory(model_history)
        # Lagre modell
        model_time_stamp = '{:%Y%m%d%H%M}'.format(datetime.datetime.now())
        self.model_directory = os.path.join(self.folderToSaveModels,
                                       "mlp-" + str(self.vocabSize) + "-" + str(self.maxSequenceLength) + "-" + str(
                                           self.epochs) + "-" + str(model_time_stamp))
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)

        save_model_path = os.path.join(self.model_directory, "model.bin")
        model.save(save_model_path)

        time_elapsed = time.time() - start_time
        # Skrive nøkkelparametere til tekstfil
        utils_nb.log_model_stats(self.model_directory, self.trainingSetPath, self.x_train
                              , num_classes, self.vocabSize, self.maxSequenceLength
                              , self.epochs, time_elapsed, save_model_path,
                              self.lossModel, self.vectorizationType, self.validationSplit, word2vec=None)

        # Lagre tokenizer
        with open(self.model_directory + '/tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        # Lagre label_indexes
        with open(self.model_directory + '/label_indexes.pickle', 'wb') as handle:
            pickle.dump(labels_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("Modell ferdig trent og lagret i " + self.model_directory)


    def fasttextTrain2mlp(self, FASTTEXT_TRAIN_FILE, MAX_SEQUENCE_LENGTH, VOCAB_SIZE, VECTORIZATION_TYPE, minNumArticlesPerDewey):
        '''Converting training_set from fasttext format to MLP-format'''

        corpus_df = utils_nb.get_articles_from_folder(FASTTEXT_TRAIN_FILE)

        corpus_df = corpus_df.groupby('dewey')['text', 'file_name', 'dewey'].filter(lambda x: len(x) >= minNumArticlesPerDewey)
        if self.strictArticleSelection:
           corpus_df = utils_nb.getStrictArticleSelection(corpus_df, self.minNumArticlesPerDewey)

        print(corpus_df.describe())

        y_train = corpus_df['dewey']
        x_train = corpus_df['text']
        print(len(y_train))
        print(len(x_train))
        labels_index = {}
        labels = []
        for dewey in set(y_train):
            label_id = len(labels_index)
            labels_index[dewey] = label_id
        for dewey in y_train:
            labels.append(labels_index[dewey])
        print("length of labels indexes: {} ".format(len(labels_index)))
        # print(labels_index)
        print("Length of labels:{}".format(len(labels)))
        num_classes = len(set(y_train))
        # Preparing_training_set
        tokenizer = Tokenizer(num_words=VOCAB_SIZE)
        tokenizer.fit_on_texts(x_train)
        sequences = tokenizer.texts_to_sequences(x_train)
        sequence_matrix = tokenizer.sequences_to_matrix(sequences, mode=VECTORIZATION_TYPE)

        data = pad_sequences(sequence_matrix, maxlen=MAX_SEQUENCE_LENGTH)

        labels = to_categorical(np.asarray(labels))

        print(labels.shape)

        x_train = data
        y_train = labels

        return x_train, y_train, tokenizer, num_classes, labels_index  # x_test, y_test, num_classes

    def predict(self, TEST_SET):
        # TEST_SET = deweys_and_texts
        '''Test module for MLP'''
        # format of test_set [[dewey][text_split1, text_split2,text_split3]]
        # Loading model
        k_output_labels = self.kPreds
        model = load_model(os.path.join(self.model_directory, 'model.bin'))


        ## TO-DO: PAKK INN ALLE DISSE FILINNHENTINGENE I EN METODE. BÅDE HER OG I CNN
        # loading tokenizer
        with open(os.path.join(self.model_directory, "tokenizer.pickle"), 'rb') as handle:
            tokenizer = pickle.load(handle)
        # loading label indexes
        with open(os.path.join(self.model_directory, "label_indexes.pickle"), 'rb') as handle:
            labels_index = pickle.load(handle)

        # Loading parameters like max_sequence_length, vocabulary_size and vectorization_type
        with open(os.path.join(self.model_directory, "model_stats"), 'r') as params_file:
            params_data = params_file.read()

        re_max_seq_length = re.search('length:(.+?)\n', params_data)
        if re_max_seq_length:
            self.maxSequenceLength = int(re_max_seq_length.group(1))
            print("Max sequence length:{}".format(self.maxSequenceLength))
        re_vocab_size = re.search('size:(.+?)\n', params_data)
        if re_vocab_size:
            self.vocabSize = int(re_vocab_size.group(1))
            print("Vocabulary size: {}".format(self.vocabSize))

        re_vectorization_type = re.search('type:(.+?)\n', params_data)
        if re_vectorization_type:
            self.vectorizationType = re_vectorization_type.group(1)
            print("This utilizes the vectorization: {}".format(str(self.vectorizationType)))

        # if isMajority_rule == True:
        #     predictions, test_accuracy = self.mlp_majority_rule_test(test_set_dewey=TEST_SET, MODEL=model,
        #                                                         MAX_SEQUENCE_LENGTH=self.maxSequenceLength,
        #                                                         TRAIN_TOKENIZER=tokenizer,
        #                                                         LABEL_INDEX_VECTOR=labels_index,
        #                                                         VECTORIZATION_TYPE=self.vectorizationType,
        #                                                         k_output_labels=k_output_labels)
        #
        # else:
        x_test, y_test = self.fasttextTest2mlp(TEST_SET, self.maxSequenceLength, tokenizer, labels_index,
                                          self.vectorizationType)
        test_score,self.accuracy = self.evaluation(model,x_test,y_test, VERBOSE = 1)

        self.predictions = utils_nb.prediction(model, x_test, k_output_labels, labels_index)
        gc.collect()


        # Writing results to txt-file.
        #with open(os.path.join(self.model_directory, "result.txt"), 'a') as result_file:
        #    result_file.write('Test_accuracy:' + str(test_accuracy) + '\n\n')
        #return predictions


    def run_evaluation(self):
        super(mlp, self).get_predictions(self.predictions, self.correct_deweys)
        super(mlp, self).evaluate_prediction()


    def printPredictionsAndAccuracy(self):
        print(self.predictions)
        print(self.accuracy)
    def fasttextTest2mlp(self ,fasttext_test_file, max_sequence_length, train_tokenizer, label_index_vector,
                         vectorization_type):
        ''' Preparing test data for MLP training'''
        test_corpus_df = utils_nb.get_articles_from_folder(fasttext_test_file)
        self.x_test = test_corpus_df['text']
        self.y_test = test_corpus_df['dewey']
        #self.correct_deweys = test_corpus_df['dewey'].values
        validDeweys = utils_nb.findValidDeweysFromTrain(self.y_test, label_index_vector)

        test_corpus_df = test_corpus_df.loc[test_corpus_df['dewey'].isin(validDeweys)]
        #print(test_corpus_df.describe())

        self.y_test = test_corpus_df['dewey']
        self.x_test = test_corpus_df['text']
        self.correct_deweys = self.y_test.values

        test_labels = []
        for dewey in self.y_test:
            test_labels.append(label_index_vector[dewey.strip()])

        test_sequences = train_tokenizer.texts_to_sequences(self.x_test)
        test_sequence_matrix = train_tokenizer.sequences_to_matrix(test_sequences, mode=vectorization_type)

        x_test = pad_sequences(test_sequence_matrix, maxlen=max_sequence_length)
        y_test = to_categorical(np.asarray(test_labels))

        return x_test, y_test

    def mlp_majority_rule_test(self, test_set_dewey, MODEL, MAX_SEQUENCE_LENGTH, TRAIN_TOKENIZER,
                               LABEL_INDEX_VECTOR
                               , VECTORIZATION_TYPE, k_output_labels):
        total_preds = []
        y_test_total = []
        one_pred = []
        for i in range(len(test_set_dewey)):

            dewey = test_set_dewey[i][0]

            texts = test_set_dewey[i][1]
            dewey_label_index = LABEL_INDEX_VECTOR[dewey.strip()]
            y_test = []
            new_texts = []

            for j in range(0, len(texts)):
                y_test.append(dewey_label_index)
                new_texts.append(' '.join(texts[j]))

            test_sequences = TRAIN_TOKENIZER.texts_to_sequences(new_texts)
            test_sequences_matrix = TRAIN_TOKENIZER.sequences_to_matrix(test_sequences, mode=VECTORIZATION_TYPE)
            x_test = pad_sequences(test_sequences_matrix, maxlen=MAX_SEQUENCE_LENGTH)
            y_test = to_categorical(np.asarray(y_test))

            predictions = utils_nb.prediction(MODEL, x_test, k_output_labels, LABEL_INDEX_VECTOR)
            y_test_total.append(dewey)
            majority_rule_preds = evaluator2.majority_rule(predictions, k_output_labels)
            total_preds.append(majority_rule_preds)
            one_pred.append(majority_rule_preds[0])

        accuracy = accuracy_score(y_test_total, one_pred)
        return total_preds, accuracy

    def evaluation(self, MODEL, X_TEST, Y_TEST, VERBOSE):
        '''Evaluates model. Return accuracy and score'''
        score = MODEL.evaluate(X_TEST, Y_TEST, VERBOSE)
        test_score = score[0]
        test_accuracy = score[1]

        return test_score, test_accuracy

    #def resultToLog(self, filepath, modelConfigs)
    def printResultToLog(self, filepath):
        super(mlp,self).resultToLog(filepath ,self.__config)

# if __name__ == '__main__':
#     model = mlp("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/mlp.yml")
#     model.fit()
#     model.predict("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/data_set/test_fredag_mlp/test_fredag_mlp_test",
#                   3,False)
#
#     print(model.config)
#     #model.train_mlp()
#     #model.train_mlp()
