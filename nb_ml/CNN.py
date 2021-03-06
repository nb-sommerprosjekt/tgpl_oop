from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Input, Conv1D, MaxPooling1D, Flatten, Dense
import numpy as np
import gensim
from keras.layers import Embedding
from keras.models import Model, load_model
import datetime
import time
#import MLP
import os
import pickle
#import evaluator2
import utils_nb
import yaml
from keras import backend as K
K.set_image_dim_ordering('tf')
import gc
from evaluator import evaluator

class cnn(evaluator):
    def __init__(self, pathToConfigFile):
        self.__config = {}
        self.load_config(pathToConfigFile)
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.correct_deweys = None
        self.embedding_matrix = []
        self.predictions = []
        self.accuracy = None
        super(cnn, self).__init__(self.evaluatorConfigPath)
    def load_config(self, pathToConfigFile):
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
        self.w2vPath = self.__config["w2vPath"]
        self.embeddingDim = self.__config["embeddingDimensions"]
        self.minNumArticlesPerDewey = self.__config["minNumArticlesPerDewey"]
        self.kPreds = self.__config["kPreds"]
        self.evaluatorConfigPath = self.__config["evaluatorConfigPath"]
        self.strictArticleSelection = self.__config["strictArticleSelection"]
    def fit(self): #EPOCHS, FOLDER_TO_SAVE_MODEL, loss_model,
                  #VALIDATION_SPLIT, word2vec_file_name):
        '''Training embedded cnn model'''
        start_time = time.time()


        self.x_train, self.y_train, word_index, labels_index, tokenizer, num_classes = self.fasttextTrain2CNN(training_set=self.trainingSetPath,
                                                                                               max_sequence_length=self.maxSequenceLength,
                                                                                               vocab_size= self.vocabSize, minNumArticlesPerDewey= self.minNumArticlesPerDewey)
        self.embedding_matrix = self.create_embedding_matrix(self.w2vPath, word_index, self.embeddingDim)
        print(self.embedding_matrix.shape)
        #
        sequence_input = Input(shape = (self.maxSequenceLength,), dtype = 'int32')
        embedding_layer = Embedding(len(word_index) + 1,
                                    self.embeddingDim,
                                    weights=[self.embedding_matrix],
                                    input_length=self.maxSequenceLength,
                                    trainable=False)
        embedded_sequences = embedding_layer(sequence_input)

        x = Conv1D(128, 5, activation = 'relu')(embedded_sequences)
        x = MaxPooling1D(5, )(x)
        x = Conv1D(128, 5, activation = 'relu')(x)
        x = MaxPooling1D(5)(x)
        x = Conv1D(128,5, activation = 'relu')(x)
        x = MaxPooling1D(15)(x)
        x = Flatten()(x)
        x = Dense(128, activation = 'relu')(x)
        #
        preds = Dense(len(labels_index), activation='softmax')(x)
        #
        model = Model(sequence_input, preds)
        model.compile(loss=self.lossModel,
                       optimizer="rmsprop",
                       metrics=['acc'])

        cnn_model = model.fit(self.x_train, self.y_train,
                   batch_size=self.batchSize,
                   epochs=self.epochs,
                  validation_split= self.validationSplit
                   )
        # list all data in history
        #utils_nb.plotTrainHistory(cnn_model)
        #print(history.history.keys())
        # summarize history for accuracy



        ### Fix for non-heurestic error:
        # https://github.com/tensorflow/tensorflow/issues/3388
        gc.collect()
        ###################


        time_stamp = '{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())
        if not os.path.exists(self.folderToSaveModels):
             os.makedirs(self.folderToSaveModels)
        self.modelDir = os.path.join(self.folderToSaveModels, "cnn-" + str(self.vocabSize) + "-" + str(self.maxSequenceLength)
                                       + "-" + str(self.epochs) + "-" + str(time_stamp))
        if not os.path.exists(self.modelDir):
            os.makedirs(self.modelDir)
        # #folder_to_save_model= MODEL_DIRECTORY+str(VOCAB_SIZE)+"-"+str(MAX_SEQUENCE_LENGTH)+"-"+str(EPOCHS)+"-"+str(time_stamp)
        #
        timeElapsed = time.time() - start_time
        #
        #
        save_model_path = self.modelDir+"/model.bin"
        utils_nb.log_model_stats(model_directory = self.modelDir, training_set_name = self.trainingSetPath
                                 , training_set = self.x_train, num_classes = num_classes, vocab_size = self.vocabSize,
                                 max_sequence_length= self.maxSequenceLength, epochs=self.epochs, time_elapsed = timeElapsed,
                                 path_to_model= save_model_path, loss_model =self.lossModel, vectorization_type= None,
                                 validation_split = self.validationSplit, word2vec = self.w2vPath)

        # #Saving model
        model.save(save_model_path)
        print("modell er nå lagret i folder: {}".format(self.modelDir))

        #Saving tokenizer
        with open(self.modelDir+'/tokenizer.pickle', 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #Saving indexes
        with open(self.modelDir+'/label_indexes.pickle', 'wb') as handle:
            pickle.dump(labels_index, handle, protocol=pickle.HIGHEST_PROTOCOL)



    def fasttextTrain2CNN(self, training_set, max_sequence_length, vocab_size, minNumArticlesPerDewey):
        '''Transforming training set from fasttext format to CNN format.'''
        corpus_df = utils_nb.get_articles_from_folder(training_set)
        ###Filtering articles by frequency of articles per dewey
        corpus_df = corpus_df.groupby('dewey')['text', 'file_name', 'dewey'].filter(lambda x: len(x) >= minNumArticlesPerDewey)
        if self.strictArticleSelection:
           corpus_df = utils_nb.getStrictArticleSelection(corpus_df, self.minNumArticlesPerDewey)
        print(corpus_df.describe())

        self.y_train = corpus_df['dewey'].values
        self.x_train = corpus_df['text'].values
        print(len(self.y_train))
        print(len(self.x_train))
        labels_index = {}
        labels = []
        for dewey in set(self.y_train):
            label_id = len(labels_index)
            labels_index[dewey] = label_id
        for dewey in self.y_train:
            labels.append(labels_index[dewey])

        num_classes = len(set(corpus_df['dewey'].values))

        tokenizer = Tokenizer(num_words= vocab_size)
        tokenizer.fit_on_texts(self.x_train)
        sequences = tokenizer.texts_to_sequences(self.x_train)

        #print(sequences)
        word_index = tokenizer.word_index
        print('Found %s unique tokens.' % len(word_index))

        x_train = pad_sequences(sequences, maxlen=max_sequence_length)

        y_train = to_categorical(np.asarray(labels))

        return x_train, y_train, word_index, labels_index, tokenizer, num_classes

    def create_embedding_matrix(self, word2vecModel,word_index, embeddingDim):
        '''Creating embedding matrix from words in vocabulary'''
        w2v_model = gensim.models.Doc2Vec.load(word2vecModel)
        embedding_matrix = np.zeros((len(word_index) + 1, embeddingDim))
        j=0
        k=0
        for word, i in word_index.items():
            k = k+1
            try:
                if w2v_model[word] is not None:
                # words not found in embedding index will be all-zeros.

                    embedding_matrix[i] = w2v_model[word]
            except KeyError:
                j=j+1
                continue
        return embedding_matrix


    def predict(self, test_set):
        '''Test module for CNN'''
        test_corpus_df = utils_nb.get_articles_from_folder(test_set)
        k_top_labels = self.kPreds
        #Loading model


        model = load_model(self.modelDir+'/model.bin')

        # loading tokenizer
        with open(self.modelDir+'/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        # loading label indexes
        with open(self.modelDir + '/label_indexes.pickle', 'rb') as handle:
            labels_index = pickle.load(handle)

        # Loading parameters like max_sequence_length, vocabulary_size and vectorization_type
        # with open(self.modelDir+'/model_stats', 'r') as params_file:
        #     params_data = params_file.read()
        #
        # re_max_seq_length = re.search('length:(.+?)\n', params_data)
        # if re_max_seq_length:
        #         self.maxSequenceLength = int(re_max_seq_length.group(1))
        #         print("Max sequence length: {}".format(MAX_SEQUENCE_LENGTH))
        # re_vocab_size = re.search('size:(.+?)\n', params_data)
        # if re_vocab_size:
        #     vocab_size = int(re_vocab_size.group(1))
        #     print("The vocabulary size: {}".format(vocab_size))
        # if isMajority_rule == True:
        #     predictions, test_accuracy = cnn_majority_rule_test(test_set_dewey=test_set, MODEL=model,
        #                                                         MAX_SEQUENCE_LENGTH = MAX_SEQUENCE_LENGTH,
        #                                                         TRAIN_TOKENIZER = tokenizer, LABEL_INDEX_VECTOR = labels_index,
        #                                                         k_output_labels=k_top_labels)


        # test_labels = []
        # valid_deweys = set()
        # # Finding valid deweys based on training set
        # for dewey in self.y_test:
        #      ##If statement to ensure that you have the same deweys in
        #      if labels_index[dewey]:
        #         #test_labels.append(labels_index[dewey])
        #         valid_deweys.update(dewey)
        self.y_test = test_corpus_df['dewey']
        self.x_test = test_corpus_df['text']

        validDeweys = utils_nb.findValidDeweysFromTrain(self.y_test, labels_index)

        #test_corpus_df = test_corpus_df[test_corpus_df['dewey'].isin(validDeweys)]
        test_corpus_df = test_corpus_df.loc[test_corpus_df['dewey'].isin(validDeweys)]
        print(test_corpus_df.describe())

        self.y_test = test_corpus_df['dewey']
        self.x_test = test_corpus_df['text']
        self.correct_deweys = self.y_test.values
        test_labels = []
        for dewey in self.y_test:
                test_labels.append(labels_index[dewey])

        test_sequences = tokenizer.texts_to_sequences(self.x_test)
        test_word_index = tokenizer.word_index
        self.x_test = pad_sequences(test_sequences, maxlen=self.maxSequenceLength)

        self.y_test = to_categorical(test_labels)

        test_score, self.accuracy = model.evaluate(self.x_test, self.y_test, batch_size= self.batchSize, verbose=1)
        self.predictions = utils_nb.prediction(model, self.x_test, k_top_labels, labels_index)

        #Writing results to txt-file.
        with open(self.modelDir+"/result.txt",'a') as result_file:
            result_file.write('test_set:'+test_set+'\n'+
                              #'Test_score:'+ str(test_score)+ '\n'
                              'Test_accuracy:' + str(self.accuracy)+'\n\n')
    def run_evaluation(self):
        super(cnn, self).get_predictions(self.predictions, self.correct_deweys)
        super(cnn, self).evaluate_prediction()

    def printResultToLog(self, filepath):
        super(cnn,self).resultToLog(filepath ,self.__config)

    def printPredictionsAndAccuracy(self):
        print(self.predictions)
        print(self.accuracy)

#if __name__ == '__main__':
    # test = cnn("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/cnn.yml")
    # test.fit()
    # test.predict("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/data_set/test_fredag_mlp/test_fredag_mlp_test", 3)
    # print(test.modelDir)

#     vocab_vector = [5000]
#     sequence_length_vector = [5000]
#     epoch_vector = [1]
#     run_cnn_tests(TRAINING_SET= "corpus_w_wiki/data_set_100/combined100_training", TEST_SET= "corpus_w_wiki/data_set_100/100_test", VOCAB_VECTOR=vocab_vector
#                    , SEQUENCE_LENGTH_VECTOR= sequence_length_vector,EPOCHS= epoch_vector, FOLDER_TO_SAVE_MODEL = "cnn/",
#                    LOSS_MODEL= "categorical_crossentropy", validation_split= 'None', word2vec_model ="w2v_tgc/full.bin", k_top_labels = 5)

    #cnn_pred("corpus_w_wiki/data_set_100/100_test", 'cnn/cnn-5000-5000-10-20171110130602', 5)