import sys
sys.path.append('/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/nb_ml')
from nb_ml import CNN, MLP, preprocessing, logreg


# ### Code for running CNN training and Predictio
# test = CNN.cnn("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/cnn.yml")
# test.fit()
# test.predict("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/data_set/test_fredag_mlp/test_fredag_mlp_test")
# test.get_predictions(test.predictions, test.correct_deweys)
# test.evaluate_prediction()
# test.resultToLog("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/cnnlogTest.txt")


#test.printPredictionsAndAccuracy()
#
## Code for running MLP training and Prediction
model = MLP.mlp("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/mlp.yml")
print(MLP.config)
#model.fit()
#model.predict("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/data_set/test_fredag_mlp/test_fredag_mlp_test", False)
#model.get_predictions(model.predictions, model.correct_deweys)
#model.evaluate_prediction()
#model.resultToLog("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/logTest.txt")
# model.printKeyMetrics()


# ### Code for running logistic regression training and prediction
# test = logreg.logReg("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/logreg.yml")
# test.fit_LogReg()
# test.predict()
#
# test.get_predictions(test.predictions, test.correct_deweys)
# test.evaluate_prediction()
# test.printKeyMetrics()
#test.printPredictionsAndAccuracy()
#print(test.correct_deweys)