import yaml
import time
import os
import MLP


class test_MLP:
	def __init__(self,pathToConfig):
		self.config = {}
		self.load_config(pathToConfig)

		# self.trainingSetPath=""
		# self.vocabSize =0
		# self.maxSequenceLength = 0
		# self.vectorizationType = None
		# self.epochs=0
		# self.minNumArticlesPerDewey = 0

	def load_config(self,pathToConfigFile):

		with open(pathToConfigFile, "r") as file:
			self.__config = yaml.load(file)
		self.configFolder= self.__config["configFolder"]
		self.logFolder= self.__config["logFolder"]
		self.trainingSetPath= self.__config["trainingSetPath"]
		self.testSetPath= self.__config["testSetPath"]
		self.vocabSizeVector = self.__config["vocabSizeVector"]
		self.maxSequenceLengthVector = self.__config["maxSequenceLengthVector"]

		self.batchSize = self.__config["batchSize"]
		self.vectorizationTypeVector = self.__config["vectorizationTypeVector"]
		self.epochsVector = self.__config["epochsVector"]
		self.validationSplit = self.__config["validationSplit"]
		self.folderToSaveModels = self.__config["folderToSaveModels"]
		self.modelDir = None
		self.lossModel = self.__config["lossModel"]
		self.minNumArticlesPerDeweyVector = self.__config["minNumArticlesPerDeweyVector"]
		self.kPreds = self.__config["kPreds"]
		self.evaluatorConfigPath = self.__config["evaluatorConfigPath"]
		self.strictArticleSelection = self.config["strictArticleSelection"]


	def run_tests(self):
		count=0
		for vocabSize in self.vocabSizeVector:
			for maxSequenceLength in self.maxSequenceLengthVector:
				for vectorizationType in self.vectorizationTypeVector:
					for epochs in self.epochsVector:
						for minNumArticlesPerDewey in self.minNumArticlesPerDeweyVector:
							new_configFile,run_name= self.create_config_file(vocabSize,maxSequenceLength,vectorizationType,epochs,minNumArticlesPerDewey)
							tid = time.time()
							count += 1
							run_length=len(self.vocabSizeVector)*len(self.maxSequenceLengthVector)*len(self.vectorizationTypeVector)*len(self.epochsVector)*len(self.minNumArticlesPerDeweyVector)
							print("Gjør test nr {} av {} : ".format(count, run_length))
							mlp_model = MLP.mlp(new_configFile)
							mlp_model.fit()
							mlp_model.predict(self.testSetPath)
							mlp_model.get_predictions(mlp_model.predictions, mlp_model.correct_deweys)
							mlp_model.evaluate_prediction()
							new_logPath=os.path.join(self.logFolder,run_name+".log")
							mlp_model.printResultToLog(new_logPath)
							print("Det tok {} \n".format(time.time() - tid))


								

	def create_config_file(self,vocabSize,maxSequenceLength,vectorizationType,epochs,minNumArticlesPerDewey):
		new_config={}

		#From vectors. Different in every config-file
		new_config["vocabSize"]=vocabSize
		new_config["maxSequenceLength"]=maxSequenceLength
		new_config["vectorizationType"]=vectorizationType
		new_config["epochs"]=epochs
		new_config["minNumArticlesPerDewey"]=minNumArticlesPerDewey

		#The same for every run
		new_config["trainingSetPath"] = self.trainingSetPath
		new_config["batchSize"]=self.batchSize
		new_config["validationSplit"]=self.validationSplit
		new_config["folderToSaveModels"]=self.folderToSaveModels
		new_config["modelDir"]=self.modelDir
		new_config["lossModel"]=self.lossModel
		new_config["kPreds"]=self.kPreds
		new_config["evaluatorConfigPath"]=self.evaluatorConfigPath
		new_config["strictArticleSelection"] = self.strictArticleSelection

		run_name="MLP-vocab-{}-maxSequenceLength-{}-vectorizationType-{}-epochs-{}-time-{}".format(vocabSize,maxSequenceLength,vectorizationType,epochs,time.time())
		new_configPath=os.path.join(self.configFolder,run_name+".yml")
		with open(new_configPath, 'w') as yaml_file:
			yaml.dump(new_config, yaml_file, default_flow_style=False)
		return new_configPath,run_name

