from linearRegressionAlgoritm import LinearRegression
import torch
from creatingTensors import CreatingTensors
from separateData import SeparateData

class EvaluateModel():
    def __init__(self):
        self.linearRegression = LinearRegression()
        self.tensors = CreatingTensors()
        self.separatedData = SeparateData()

    def evaluatingModel(self):
        self.linearRegression.eval()
        with torch.no_grad():
            funcPred_test = self.linearRegression(self.tensors.tensorTestX())
            print("\nPrevisão: ")
            for i in range(len(self.separatedData.separateTrainTest()[1])):
                print(f"Entrada: {self.separatedData.separateTrainTest()[1][i]} -- Real: {self.separatedData.separateTrainTest()[3][i][0]} | Previsto: {funcPred_test[i].item():.2f}")

    def showBiasWeights(self):
        for nome, parametro in self.linearRegression.named_parameters():
            print(f"{nome}: {parametro.data.numpy()}")
