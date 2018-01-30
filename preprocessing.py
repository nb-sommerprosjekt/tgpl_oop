import sys
sys.path.append('/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/nb_ml')
from nb_ml import preprocessing


# # ## Making preprocessing object
# train_test = preprocessing.Corpus("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/preprocess.yml")
#
#
#
# ## Start filtering of data
# train_test.preprocess()
#
# ## Split data into training and test
# train_test.split_to_training_and_test()

# #Add wikipedia data to corpus
# train_test.preprocess_wiki()
# train_test.add_wiki_to_training()

# Split articles
#train_test.split_articles()

## Create fake articles
test = preprocessing.dataAugmention()
test.getConfig("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/preprocess20.yml",
               "/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/data_set/tgcForOptimization/tgcForOptimization_training", "/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/new_artificial_test20")
test.get_articles()
test.create_fake_corpus()
test.copyArtificialFolderIntoCorpus()
#
# ## Create fake articles
# test = preprocessing.dataAugmention()
# test.getConfig("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/preprocess15.yml",
#                "/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/data_set/tgcForOptimization/tgcForOptimization_training", "/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/new_artificial_test15")
# test.get_articles()
# test.create_fake_corpus()
# test.copyArtificialFolderIntoCorpus()
#
# ## Create fake articles
# test = preprocessing.dataAugmention()
# test.getConfig("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/preprocess30.yml",
#                "/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/data_set/tgcForOptimization/tgcForOptimization_training", "/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/new_artificial_test30")
# test.get_articles()
# test.create_fake_corpus()
# test.copyArtificialFolderIntoCorpus()
