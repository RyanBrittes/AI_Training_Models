from trainingModel import TrainingEvaluateModel
from evaluateModel import EvaluateModel

class App():
    def __init__(self):
        self.training = TrainingEvaluateModel()

    def showModel(self):
        self.training.iteratorTrain()

A = App()

A.showModel()