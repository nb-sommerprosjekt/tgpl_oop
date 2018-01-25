import sys
sys.path.append('/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/nb_ml')
from nb_ml import CNN, MLP, preprocessing, logreg, fast_text

#
# ## Code for running CNN training and Prediction
# test = CNN.cnn("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/cnn.yml")
# test.fit()
# test.predict("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/data_set/tgcForOptimization/tgcForOptimization_test")
# test.run_evaluation()
# test.printResultToLog("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/cnnTestStrict.txt")


#test.printPredictionsAndAccuracy()
#
## Code for running MLP training and Prediction
# #
# model = MLP.mlp("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/mlp.yml")
# model.fit()
# model.predict("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/data_set/tgcForOptimization/tgcForOptimization_test")
# model.run_evaluation()
# model.printResultToLog("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/mlpTestStrict50.txt")






#
# # ### Code for running logistic regression training and prediction
test = logreg.logReg("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/logreg.yml")
test.fit()
#test.findFeatureImportance()
test.predict()
test.run_evaluation()
test.printResultToLog("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/logRes_testStrict.txt")






# ft_test = fast_text.fast_text("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/fasttext.yml")
# ft_test.fit()
# ft_test.predict("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/data_set/tgcForOptimization/tgcForOptimization_test")
# ft_test.run_evaluation()
# ft_test.printResultToLog("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/fasttext_test_strictdewey.txt")
