import yaml

class evaluator():

    def __init__(self, pathToConfigFile):
        self.load_config(pathToConfigFile = pathToConfigFile)

    def load_config(self, pathToConfigFile):
        with open(pathToConfigFile, "r") as file:
            self.config = yaml.load(file)