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
# Code for running MLP training and Prediction
#
# model = MLP.mlp("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/test_strict/mlp/mlp_opt_strict50.yml")
# model.fit()
# model.predict("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/data_set/tgcForOptimization/tgcForOptimization_test")
# model.run_evaluation()
# model.printResultToLog("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/test_strict/mlp/logs/mlp_opt_strict50.log")
#
# model = MLP.mlp("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/test_strict/mlp/mlp_opt_strict100.yml")
# model.fit()
# model.predict("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/data_set/tgcForOptimization/tgcForOptimization_test")
# model.run_evaluation()
# model.printResultToLog("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/test_strict/mlp/logs/mlp_opt_strict100.log")
#
# model = MLP.mlp("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/test_strict/mlp/mlp_opt_strict150.yml")
# model.fit()
# model.predict("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/data_set/tgcForOptimization/tgcForOptimization_test")
# model.run_evaluation()
# model.printResultToLog("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/test_strict/mlp/logs/mlp_opt_strict150.log")

# model = MLP.mlp("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/test_strict/mlp/mlp_opt_strict200.yml")
# model.fit()
# model.predict("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/data_set/tgcForOptimization/tgcForOptimization_test")
# model.run_evaluation()
# model.printResultToLog("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/test_strict/mlp/logs/mlp_opt_strict200.log")

# model = MLP.mlp("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/test_strict/mlp/mlp_opt_strict300.yml")
# model.fit()
# model.predict("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/data_set/tgcForOptimization/tgcForOptimization_test")
# model.run_evaluation()
# model.printResultToLog("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/test_strict/mlp/logs/mlp_opt_strict300.log")





#
# # # ### Code for running logistic regression training and prediction
test = logreg.logReg("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/test_strict/logres/logreg_opt_strict50.yml")
test.fit()
test.predict()
test.run_evaluation()
test.printResultToLog("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/test_strict/logres/logs/svc_opt_strict50.log")
#
# test = logreg.logReg("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/test_strict/logres/logreg_opt_strict100.yml")
# test.fit()
# test.predict()
# test.run_evaluation()
# test.printResultToLog("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/test_strict/logres/logs/svc_opt_strict100.log")
#
# test = logreg.logReg("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/test_strict/logres/logreg_opt_strict150.yml")
# test.fit()
# test.predict()
# test.run_evaluation()
# test.printResultToLog("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/test_strict/logres/logs/svc_opt_strict150.log")
#
# test = logreg.logReg("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/test_strict/logres/logres_opt_strict200.yml")
# test.fit()
# test.predict()
# test.run_evaluation()
# test.printResultToLog("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/test_strict/logres/logs/svc_opt_strict200.log")
#
# test = logreg.logReg("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/test_strict/logres/logreg_opt_strict300.yml")
# test.fit()
# test.predict()
# test.run_evaluation()
# test.printResultToLog("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/test_strict/logres/logs/save_vectorizer_test.log")




# ft_test = fast_text.fast_text("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/test_strict/fasttext/fasttext_strict_opt50.yml")
# ft_test.fit()
# ft_test.predict("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/data_set/tgcForOptimization/tgcForOptimization_test")
# ft_test.run_evaluation()
# ft_test.printResultToLog("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/test_strict/fasttext/logs/fasttext_strict_opt50.log")
#
#
# ft_test = fast_text.fast_text("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/test_strict/fasttext/fasttext_strict_opt100.yml")
# ft_test.fit()
# ft_test.predict("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/data_set/tgcForOptimization/tgcForOptimization_test")
# ft_test.run_evaluation()
# ft_test.printResultToLog("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/test_strict/fasttext/logs/fasttext_strict_opt100.log")
#
# ft_test = fast_text.fast_text("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/test_strict/fasttext/fasttext_strict_opt150.yml")
# ft_test.fit()
# ft_test.predict("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/data_set/tgcForOptimization/tgcForOptimization_test")
# ft_test.run_evaluation()
# ft_test.printResultToLog("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/test_strict/fasttext/logs/fasttext_strict_opt150.log")
#
# ft_test = fast_text.fast_text("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/test_strict/fasttext/fasttext_strict_opt200.yml")
# ft_test.fit()
# ft_test.predict("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/data_set/tgcForOptimization/tgcForOptimization_test")
# ft_test.run_evaluation()
# ft_test.printResultToLog("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/test_strict/fasttext/logs/fasttext_strict_opt200.log")
#
# ft_test = fast_text.fast_text("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/test_strict/fasttext/fasttext_strict_opt300.yml")
# ft_test.fit()
# ft_test.predict("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/data_set/tgcForOptimization/tgcForOptimization_test")
# ft_test.run_evaluation()
# ft_test.printResultToLog("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/test_strict/fasttext/logs/fasttext_strict_opt300.log")

