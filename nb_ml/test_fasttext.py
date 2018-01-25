import yaml
import time
import os
import fast_text
import datetime


class test_fasttext:
	def __init__(self, pathToConfigFile):
		self.__config = {}
		self.load_config(pathToConfigFile)


	def load_config(self, pathToConfigFile):
		with open(pathToConfigFile, "r") as file:
			self.__config = yaml.load(file)

		#meta-variables:
		self.configFolder= self.__config["configFolder"]
		self.testSetPath = self.__config["testSetPath"]
		self.logFolder= self.__config["logFolder"]

		#parameters that might change during the testing:
		self.epochsVector = self.__config["epochsVector"]
		self.learningRateVector = self.__config["learningRateVector"]
		self.lrUpdateVector = self.__config["lrUpdateVector"]
		self.lossFunctionVector = self.__config["lossFunctionVector"]
		self.wikiVecVector = self.__config["wikiVecVector"]
		self.wordWindowVector = self.__config["wordWindowVector"]
		self.minNumArticlesPerDeweyVector = self.__config["minNumArticlesPerDeweyVector"]

		#The same for every Fasttext-model created:
		self.evaluatorConfigPath = self.__config["evaluatorConfigPath"]
		self.trainingSetPath = self.__config["trainingSetPath"]
		self.wikiPath = self.__config["wikiPath"]
		self.kPreds = self.__config["kPreds"]
		self.modelsDir = self.__config["modelsDir"]
		timestamp = '{:%Y%m%d%H%M%S}'.format(datetime.datetime.now())
		self.folderToSaveModels = self.modelsDir + "/fasttext-" + timestamp
		if not os.path.exists(self.folderToSaveModels):
			os.makedirs(self.folderToSaveModels)
		self.tmp_ft_file_path = self.folderToSaveModels + "/tmp.txt"
		self.strictArticleSelection = self.config["strictArticleSelection"]


	def run_tests(self):
		count=0
		for wikiVec in self.wikiVecVector:
			for lrUpdate in self.lrUpdateVector:
				for lossFunction in self.lossFunctionVector:
					for learningRate in self.learningRateVector:
						for wordWindow in self.wordWindowVector:
							for epochs in self.epochsVector:
								for minNumArticlesPerDewey in self.minNumArticlesPerDeweyVector:
									tid=time.time()
									run_length=len(self.wikiVecVector)*len(self.lrUpdateVector)*len(self.lossFunctionVector)*len(self.learningRateVector)*len(self.wordWindowVector)*len(self.epochsVector)*len(self.minNumArticlesPerDeweyVector)
									count += 1
									print("Gjør test nr {} av {} : ".format(count,run_length))
									new_configFile, run_name = self.create_config_file(wikiVec, lrUpdate,
									                                                   learningRate,lossFunction,wordWindow, epochs,
									                                                   minNumArticlesPerDewey)
									fasttext_model = fast_text.fast_text(new_configFile)
									fasttext_model.fit()
									fasttext_model.predict(self.testSetPath)
									fasttext_model.run_evaluation()
									new_logPath = os.path.join(self.logFolder, run_name + ".log")
									fasttext_model.printResultToLog(new_logPath)
									print("Det tok {} \n".format(time.time() - tid))

		

	def create_config_file(self, wikiVec, lrUpdate, learningRate, lossFunction,wordWindow,epochs, minNumArticlesPerDewey):
		new_config = {}

		# From vectors. Different in every config-file
		new_config["wikiVec"] = wikiVec
		new_config["lrUpdate"] = lrUpdate
		new_config["learningRate"] = learningRate
		new_config["lossFunction"] = lossFunction
		new_config["wordWindow"] = wordWindow
		new_config["epochs"] = epochs
		new_config["minNumArticlesPerDewey"] = minNumArticlesPerDewey

		# The same for every run
		new_config["trainingSetPath"] = self.trainingSetPath
		new_config["modelsDir"] = self.modelsDir
		new_config["wikiPath"] = self.wikiPath
		new_config["kPreds"] = self.kPreds
		new_config["evaluatorConfigPath"] = self.evaluatorConfigPath
		new_config["strictArticleSelection"] = self.strictArticleSelection

		run_name = "fasttext-wikiVec-{}-lrUpdate-{}-learningRate-{}-lossFunction-{}-epochs-{}-time-{}".format(wikiVec,
		                                                                                           lrUpdate,
		                                                                                      learningRate,lossFunction,
		                                                                                             epochs,
		                                                                                             time.time())
		new_configPath = os.path.join(self.configFolder, run_name + ".yml")
		with open(new_configPath, 'w') as yaml_file:
			yaml.dump(new_config, yaml_file, default_flow_style=False)
		return new_configPath, run_name

