import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def get_articles(fasttext_text_file):
    # Tar inn en textfil som er labelet på fasttext-format. Gir ut to arrays. Et med deweys og et med tekstene. [deweys],[texts]
    articles=open(fasttext_text_file,"r")
    articles=articles.readlines()
    dewey_array = []
    docs = []
    dewey_dict = {}
    for article in articles:
        dewey=article.partition(' ')[0].replace("__label__","")
        article_label_removed = article.replace("__label__"+dewey,"")
        docs.append(article_label_removed)
        dewey_array.append(dewey)

    return dewey_array, docs

def get_articles_from_folder(folder):
    ''' Tar inn en textfil som er labelet på fasttext-format. Gir ut 3 arrays. Et med navn på tekstfilene,
    et med deweys og et med tekstene. [tekst_navn][deweys],[texts]'''
    arr = os.listdir(folder)
    arr_txt = [path for path in os.listdir(folder) if path.endswith(".txt")]


    dewey_array = []
    docs = []
    for article_path in arr_txt:
            article = open(os.path.join(folder,article_path), "r")
            article = article.readlines()
            for article_content in article:
                dewey=article_content.partition(' ')[0].replace("__label__","")
                text  = article_content.replace("__label__"+dewey,"")
                docs.append(text)
                dewey_array.append(dewey[:3])
    text_names = [path.replace('.txt','') for path in arr_txt ]
    #print(len(dewey_array))
    #print(text_names)

    corpus_df= pd.DataFrame(
        {'file_name': text_names,
         'text': docs,
         'dewey': dewey_array
         })

    return corpus_df

def get_articles_from_folder_several_deweys(folder):
    ''' Tar inn en textfil som er labelet på fasttext-format. Gir ut 3 arrays. Et med navn på tekstfilene,
    et med deweys og et med tekstene. [tekst_navn][deweys],[texts]'''
    arr = os.listdir(folder)
    arr_txt = [path for path in os.listdir(folder) if path.endswith(".txt")]


    dewey_array = []
    docs = []
    for article_path in arr_txt:
            article = open(os.path.join(folder,article_path), "r")
            article = article.readlines()
            for article_content in article:
                dewey=article_content.partition(' ')[0].replace("__label__","")
                text  = article_content.replace("__label__"+dewey,"")
                docs.append(text)
                dewey_array.append(dewey.replace(".",""))
    text_names = [path.replace('.txt','') for path in arr_txt ]
    #print(len(dewey_array))
    #print(text_names)

    corpus_df= pd.DataFrame(
        {'file_name': text_names,
         'text': docs,
         'dewey': dewey_array
         })

    return corpus_df

def log_model_stats(model_directory, training_set_name, training_set,
                 num_classes,vocab_size, max_sequence_length,
                epochs, time_elapsed,
                path_to_model, loss_model, vectorization_type, validation_split, word2vec ):
    ''' Prints model parameters to log-file.'''

    time_stamp = '{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())

    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    model_stats_file = open(os.path.join(model_directory,"model_stats"), "a")

    model_stats_file.write("Time:" + str(time_stamp) + '\n'
                      + "Time elapsed: " + str(time_elapsed) + '\n'
                      + "Training set:" + training_set_name + "\n"
                      + "Vocab_size:" + str(vocab_size) + '\n'
                      + "Max_sequence_length:" + str(max_sequence_length) + '\n'
                      + "Epochs:" + str(epochs) + '\n'
                      + "Antall deweys:" + str(num_classes) + '\n'
                      + "Antall docs i treningssett:" + str(len(training_set)) + '\n'
                      + "saved_model:" + path_to_model + '\n'
                      + "loss_model:" + str(loss_model) + '\n'
                      + "vectorization_type:" + str(vectorization_type)+'\n'
                      + "Validation_split:" + str(validation_split) + '\n'
                      + "word2vec:" + str(word2vec)+'\n\n\n'
                      )
    model_stats_file.close()

def prediction(MODEL,X_TEST,k_preds, label_indexes):
    predictions = MODEL.predict(x=X_TEST)
    all_topk_labels = []
    for prediction_array in predictions:
        np_prediction = np.argsort(-prediction_array)[:k_preds]
        topk_labels = []
        for np_pred in np_prediction:

            for label_name, label_index in label_indexes.items():

                if label_index == np_pred:
                    label = label_name
                    topk_labels.append(label)

        all_topk_labels.append(topk_labels)
    return all_topk_labels



def findValidDeweysFromTrain(listOfDeweys, labelIndexDictFromTrain):
    valid_deweys = []
    # Finding valid deweys based on training set
    for dewey in listOfDeweys:
        if dewey in labelIndexDictFromTrain:
            valid_deweys.append(dewey)
    return valid_deweys



def plotTrainHistory(model):
        plt.plot(model.history['acc'])
        plt.plot(model.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(model.history['loss'])
        plt.plot(model.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()


def getStrictArticleSelection(corpus_dataframe, articlesPerDewey):
    size = articlesPerDewey  # sample size
    replace = False  # with replacement
    fn = lambda obj: obj.loc[np.random.choice(obj.index, size, replace), :]
    corpus_dataframe = corpus_dataframe.groupby('dewey', as_index=False).apply(fn)
    return corpus_dataframe