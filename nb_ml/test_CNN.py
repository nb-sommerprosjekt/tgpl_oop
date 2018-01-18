import yaml
import time
import os
import CNN


class test_CNN:
	def __init__(self, pathToConfigFile):
		self.config = {}
		self.load_config(pathToConfigFile)


	def load_config(self, pathToConfigFile):
		with open(pathToConfigFile, "r") as file:
			self.config = yaml.load(file)

		self.trainingSetPath = self.config["trainingSetPath"]
		self.vocabSizeVector = self.config["vocabSizeVector"]
		self.configFolder= self.config["configFolder"]
		self.testSetPath = self.config["testSetPath"]
		self.logFolder= self.config["logFolder"]
		self.maxSequenceLengthVector = self.config["maxSequenceLengthVector"]

		self.batchSize = self.config["batchSize"]
		self.vectorizationTypeVector = self.config["vectorizationTypeVector"]
		self.epochsVector = self.config["epochsVector"]
		self.validationSplit = self.config["validationSplit"]
		self.folderToSaveModels = self.config["folderToSaveModels"]
		self.modelDir = None
		self.lossModel = self.config["lossModel"]
		self.w2vPath = self.config["w2vPath"]
		self.embeddingDimensions = self.config["embeddingDimensions"]
		self.minNumArticlesPerDeweyVector = self.config["minNumArticlesPerDeweyVector"]
		self.kPreds = self.config["kPreds"]
		self.evaluatorConfigPath = self.config["evaluatorConfigPath"]

	def run_tests(self):
		for vocabSize in self.vocabSizeVector:
			for maxSequenceLength in self.maxSequenceLengthVector:
				for vectorizationType in self.vectorizationTypeVector:
					for epochs in self.epochsVector:
						for minNumArticlesPerDewey in self.minNumArticlesPerDeweyVector:
							new_configFile, run_name = self.create_config_file(vocabSize, maxSequenceLength,
							                                                   vectorizationType, epochs,
							                                                   minNumArticlesPerDewey)
							cnn_model = CNN.cnn(new_configFile)
							cnn_model.fit()
							cnn_model.predict(self.testSetPath)
							cnn_model.run_evaluation()
							new_logPath = os.path.join(self.logFolder, run_name + ".log")
							cnn_model.resultToLog(new_logPath)

	def create_config_file(self, vocabSize, maxSequenceLength, vectorizationType, epochs, minNumArticlesPerDewey):
		new_config = {}

		# From vectors. Different in every config-file
		new_config["vocabSize"] = vocabSize
		new_config["maxSequenceLength"] = maxSequenceLength
		new_config["vectorizationType"] = vectorizationType
		new_config["epochs"] = epochs
		new_config["minNumArticlesPerDewey"] = minNumArticlesPerDewey

		# The same for every run
		new_config["trainingSetPath"] = self.trainingSetPath
		new_config["batchSize"] = self.batchSize
		new_config["validationSplit"] = self.validationSplit
		new_config["folderToSaveModels"] = self.folderToSaveModels
		new_config["modelDir"] = self.modelDir
		new_config["lossModel"] = self.lossModel
		new_config["kPreds"] = self.kPreds
		new_config["evaluatorConfigPath"] = self.evaluatorConfigPath
		new_config["w2vPath"] = self.w2vPath
		new_config["embeddingDimensions"] = self.embeddingDimensions

		run_name = "CNN-vocab-{}-maxSequenceLength-{}-vectorizationType-{}-epochs-{}-time-{}".format(vocabSize,
		                                                                                             maxSequenceLength,
		                                                                                             vectorizationType,
		                                                                                             epochs,
		                                                                                             time.time())
		new_configPath = os.path.join(self.configFolder, run_name + ".yml")
		with open(new_configPath, 'w') as yaml_file:
			yaml.dump(new_config, yaml_file, default_flow_style=False)
		return new_configPath, run_name

