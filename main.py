import sys
sys.path.append('/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/nb_ml')
from nb_ml import CNN, MLP, preprocessing


## Making preprocessing object
train_test = preprocessing.BaseData()
train_test.load_config("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/preprocess.yml")


## Start filtering of data
train_test.preprocess()

## Split data into training and test
train_test.split_to_training_and_test()

#Add wikipedia data to corpus
train_test.preprocess_wiki()
train_test.add_wiki_to_training()

# Split articles|
train_test.split_articles()

## Create fake articles
test = preprocessing.dataAugmention()
test.getConfig("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/preprocess.yml",
               "/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/data_set/test_torsdag_mlp/test_torsdag_mlp_training_split")
test.get_articles()
test.create_fake_corpus()
test.copyArtificialFolderIntoCorpus()

### Code for running CNN training and Predictio
test = CNN.cnn("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/cnn.yml")
test.fit()
test.predict("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/data_set/test_fredag_mlp/test_fredag_mlp_test", 3)


### Code for running MLP training and Prediction
model = MLP.mlp("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/config/mlp.yml")
model.fit()
model.predict("/home/ubuntu/PycharmProjects_saved/tgpl_w_oop/data_set/test_fredag_mlp/test_fredag_mlp_test",
              3, False)