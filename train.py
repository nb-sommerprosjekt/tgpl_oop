import sys
sys.path.append('/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/nb_ml')
from nb_ml import CNN, MLP, preprocessing, logreg, fast_text

#
### Code for running CNN training and Prediction
test = CNN.cnn("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/cnn_optimized.yml")
test.fit()
test.predict("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/data_set/tgcForOptimization/tgcForOptimization_test")
test.run_evaluation()
test.printResultToLog("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/cnn_optimized100.txt")


#test.printPredictionsAndAccuracy()
#
# ## Code for running MLP training and Prediction
# #
# model = MLP.mlp("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/mlp_optimized.yml")
# model.fit()
# model.predict("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/data_set/tgcForOptimization/tgcForOptimization_test")
# model.run_evaluation()
# model.printResultToLog("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/mlpTestOptimised50.txt")

# model = MLP.mlp("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/mlp.yml")
# model.fit()
# model.predict("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/data_set/test_fredag_mlp/test_fredag_mlp_test")
# model.run_evaluation()
# model.printResultToLog("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/logTest.txt")
#model.resultToLog("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/logTest.txt")



#
# # # ### Code for running logistic regression training and prediction
# test = logreg.logReg("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/logreg.yml")
# test.fit()
# #test.findFeatureImportance()
# test.predict()
# test.run_evaluation()
# test.printResultToLog("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/logResAll.txt")


# test.get_predictions(test.predictions, test.correct_deweys)
# test.evaluate_prediction()
# test.printKeyMetrics()
# test.printPredictionsAndAccuracy()
# print(test.correct_deweys)
#
# ft_test = fast_text.fast_text("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/fasttext_optimizedparams.yml")
# ft_test.fit()
# ft_test.predict("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/data_set/tgcForOptimization/tgcForOptimization_test")
# ft_test.run_evaluation()
# ft_test.printResultToLog("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/fasttextOptimizedAll.txt")
