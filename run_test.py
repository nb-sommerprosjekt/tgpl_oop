import sys
sys.path.append('/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/nb_ml')
from nb_ml import test_MLP,test_CNN,test_fasttext


# new_test=test_MLP.test_MLP("/home/knutf/workspace/shipping_ready/tgpl_oop/config/test_mlp_sveinb.yml")
#
# new_test.run_tests()


# new_test=test_CNN.test_CNN("/home/knutf/workspace/shipping_ready/tgpl_oop/config/test_cnn.yml")
#
# new_test.run_tests()


#new_test=test_fasttext.test_fasttext("/home/knutf/workspace/shipping_ready/tgpl_oop/config/test_fasttext.yml")

#new_test.run_tests()

sveinb_test = test_MLP.test_MLP("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/test_mlp_sveinb.yml")

sveinb_test.run_tests()