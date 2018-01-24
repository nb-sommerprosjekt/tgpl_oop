import sys
sys.path.append('/home/knutf/workspace/shipping_ready/tgpl_oop/nb_ml')
from nb_ml import test_MLP,test_CNN,test_fasttext


# new_test=test_MLP.test_MLP("/home/knutf/workspace/shipping_ready/tgpl_oop/config/test_mlp.yml")
#
# new_test.run_tests()


# new_test=test_CNN.test_CNN("/home/knutf/workspace/shipping_ready/tgpl_oop/config/test_cnn.yml")
#
# new_test.run_tests()


new_test=test_fasttext.test_fasttext("/home/knutf/workspace/shipping_ready/tgpl_oop/config/test_fasttext.yml")

new_test.run_tests()