from linearRegressionAlgoritm import LinearRegression
from creatingTensors import CreatingTensors
from separateData import SeparateData
import torch.nn as nn
import torch


class TrainingEvaluateModel():
    def __init__(self):
        self.epochs = 1000
        self.linearRegression = LinearRegression()
        self.tensors = CreatingTensors()
        self.separatedData = SeparateData()
        self.optimizer = torch.optim.SGD(self.linearRegression.parameters(), lr=0.00001)
        self.pred = self.linearRegression(self.tensors.tensorTrainX())
        self.loss = nn.MSELoss()

    def iteratorTrain(self):
        for epoch in range(self.epochs):
            self.linearRegression.train()
            self.pred
            FuncLoss = self.loss(self.pred, self.tensors.tensorTrainY())

            self.optimizer.zero_grad()
            FuncLoss.backward()
            self.optimizer.step()

            if(epoch + 1) % 100 == 0:
                print(f"Período {epoch+1}/{self.epochs} -- Loss: {FuncLoss.item():.4f}")

    def evaluatingModel(self):
        self.linearRegression.eval()
        with torch.no_grad():
            funcPred_test = self.linearRegression(self.tensors.tensorTestX())
            print("\nPrevisão: ")
            for i in range(len(self.separatedData.separateTrainTest()[1])):
                print(f"Entrada: {self.separatedData.separateTrainTest()[1][i]} -- Real: {self.separatedData.separateTrainTest()[3][i][0]} | Previsto: {self.pred[i].item():.2f}")

    def showBiasWeights(self):
        for nome, parametro in self.linearRegression.named_parameters():
            print(f"{nome}: {parametro.data.numpy()}")