import sys
sys.path.append('/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/nb_ml')
from nb_ml import CNN, MLP, preprocessing, logreg, fast_text


# ### Code for running CNN training and Predictio
# test = CNN.cnn("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/cnn.yml")
# test.fit()
# test.predict("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/data_set/test_fredag_mlp/test_fredag_mlp_test")
# test.run_evaluation()
# test.resultToLog("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/cnnlogTest.txt")


#test.printPredictionsAndAccuracy()
#
## Code for running MLP training and Prediction

# model = MLP.mlp("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/mlp.yml")
# model.fit()
# model.predict("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/data_set/test_fredag_mlp/test_fredag_mlp_test", False)
# model.run_evaluation()
# model.resultToLog("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/logTest.txt")



# ### Code for running logistic regression training and prediction
test = logreg.logReg("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/logreg.yml")
test.fit()
test.predict()
test.run_evaluation()
test.resultToLog("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/logResTest.txt")
# test.get_predictions(test.predictions, test.correct_deweys)
# test.evaluate_prediction()
# test.printKeyMetrics()
# test.printPredictionsAndAccuracy()
# print(test.correct_deweys)

# ft_test = fast_text.fast_text("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/fasttext.yml")
# ft_test.fit()
# ft_test.predict("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/data_set/test_fredag_mlp/test_fredag_mlp_test")
# ft_test.run_evaluation()
# ft_test.resultToLog("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/fasttext_logTest.txt")
