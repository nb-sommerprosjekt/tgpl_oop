import yaml
import utils
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
class logReg():

    def __init__(self, pathToConfigFile):
        self.config = {}
        self.load_config(pathToConfigFile = pathToConfigFile)
        self.x_train = []
        self.y_train = []
        self.validDeweys = []
    def load_config(self, pathToConfigFile):
        with open(pathToConfigFile, "r") as file:
            self.config = yaml.load(file)
        self.training_set = self.config["training_set"]
        self.test_set = self.config["test_set"]
        self.minNumArticlesPerDewey = self.config["minNumArticlesPerDewey"]

    def fit_LogReg(self):
        print("Something will be written here")
        self.fasttext2sklearn()
        tfidf = TfidfVectorizer(norm = 'l2', min_df = 2, use_idf = True, smooth_idf= False, sublinear_tf = True, ngram_range = (1,4),
                                max_features = 50000)
        x_train_tfidf = tfidf.fit_transform(self.x_train[:100])

        test_corpus_df = utils.get_articles_from_folder(self.test_set)
        test_corpus_df = test_corpus_df.loc[test_corpus_df['dewey'].isin(self.validDeweys)]
        self.y_test = test_corpus_df['dewey']
        self.x_test = test_corpus_df['text']

        x_test_tfidf = tfidf.transform(self.x_test)

        logMod = LogisticRegression().fit(x_train_tfidf,self.y_train[:100])



    # def logReg(x_train, y_train, vectorization_type, penalty, dual, tol, C, fit_intercept, intercept_scaling,
    #            class_weight, random_state, solver, max_iter, multi_class, verbose, warm_start, n_jobs):
    #
    #     if vectorization_type == "tfidf":
    #         model = logReg_tfidf(x_train, y_train, penalty, dual, tol, C, fit_intercept,
    #                              intercept_scaling, class_weight, random_state,
    #                              solver, max_iter, multi_class,
    #                              verbose, warm_start, n_jobs)
    #     elif vectorization_type == "count":
    #         model = logReg_Count(x_train, y_train, penalty, dual, tol, C, fit_intercept,
    #                              intercept_scaling, class_weight, random_state,
    #                              solver, max_iter, multi_class,
    #                              verbose, warm_start, n_jobs)
    #     else:
    #         print("Vectorization type is not existing. Alternatives: tfidf or count")
    #         model = None
    #     return model

    # def logReg_train_and_test(x_train, y_train, x_test, y_test, vectorization_type, model_dir, k_preds=3, penalty="l2",
    #                           dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None,
    #                           random_state=None, solver="liblinear", max_iter=100, multi_class="ovr",
    #                           verbose=0, warm_start=False, n_jobs=1):
    #     if not os.path.exists(model_dir):
    #         os.makedirs(model_dir)
    #     model = logReg(x_train, y_train, vectorization_type, penalty=penalty, dual=dual,
    #                    tol=tol, C=C, fit_intercept=fit_intercept, intercept_scaling=intercept_scaling,
    #                    class_weight=class_weight,
    #                    random_state=random_state, solver=solver,
    #                    max_iter=max_iter, multi_class=multi_class, verbose=verbose, warm_start=warm_start,
    #                    n_jobs=n_jobs)
    #     model_name = "model.pickle"
    #     timestamp = '{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())
    #     save_path = model_dir + "/logReg-" + vectorization_type + timestamp
    #
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     saveSklearnModels(model, "model.pickle", save_path)
    #     model_path = save_path + "/model.pickle"
    #
    #     # testing model
    #     model_loaded = loadSklearnModel(model_path)
    #     predictions, accuracy, topN = getPredictionsAndAccuracy(x_test=x_test, y_test=y_test, model=model_loaded,
    #                                                             returnAccuracy=True, Npreds=k_preds)
    #
    #     return predictions, accuracy, topN
    #
    # def logReg_tfidf(x_train, y_train, penalty, dual, tol, C, fit_intercept, intercept_scaling, class_weight,
    #                  random_state, solver, max_iter, multi_class,
    #                  verbose, warm_start, n_jobs):
    #     logres_tfidf_model = Pipeline([("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)),
    #                                    ("logres_tfidf", LogisticRegression(penalty=penalty, dual=dual,
    #                                                                        tol=tol, C=C, fit_intercept=fit_intercept,
    #                                                                        intercept_scaling=intercept_scaling,
    #                                                                        class_weight=class_weight,
    #                                                                        random_state=random_state, solver=solver,
    #                                                                        max_iter=max_iter, multi_class=multi_class,
    #                                                                        verbose=verbose, warm_start=warm_start,
    #                                                                        n_jobs=n_jobs))])
    #     logres_tfidf_model.fit(x_train, y_train)
    #     return logres_tfidf_model
    #
    # def logReg_Count(x_train, y_train, penalty, dual, tol, C, fit_intercept,
    #                  intercept_scaling, class_weight, random_state, solver,
    #                  max_iter, multi_class, verbose, warm_start, n_jobs):
    #     logres_model = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)),
    #                              ("logres", LogisticRegression(penalty=penalty, dual=dual,
    #                                                            tol=tol, C=C, fit_intercept=fit_intercept,
    #                                                            intercept_scaling=intercept_scaling,
    #                                                            class_weight=class_weight,
    #                                                            random_state=random_state, solver=solver,
    #                                                            max_iter=max_iter, multi_class=multi_class,
    #                                                            verbose=verbose, warm_start=warm_start, n_jobs=n_jobs))])
    #     logres_model.fit(x_train, y_train)
    #     return logres_model


    def fasttext2sklearn(self):

            corpus_df = utils.get_articles_from_folder(self.training_set)
            ###Filtering articles by frequency of articles per dewey
            corpus_df = corpus_df.groupby('dewey')['text', 'file_name', 'dewey'].filter(lambda x: len(x) >= self.minNumArticlesPerDewey)
            self.y_train = corpus_df['dewey']
            self.x_train = corpus_df['text']
            self.findValidDeweysSklearn()

    def findValidDeweysSklearn(self):
        self.validDeweys = list(set(self.y_train))


if __name__ == '__main__':
    print("starting")
    test = logReg("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/logreg.yml")
    test.fit_LogReg()


