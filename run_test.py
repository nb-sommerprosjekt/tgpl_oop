import sys
sys.path.append('/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/nb_ml')
from nb_ml import test_MLP


new_test=test_MLP("/home/knutf/workspace/shipping_ready/tgpl_oop/config/test_mlp.yml")

new_test.run_tests()