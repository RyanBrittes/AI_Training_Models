#Machine learning
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

#Data
import pandas as pd
import numpy as np

#Visualization
import matplotlib.pylab as plt


#Dataset definition
dataset = pd.read_csv('/home/ryan/Documents/Python/AI/AI_Training_Models/LinearRegression/sample_employees.csv')

#Extract values and separe X and Y
valueX = dataset[['age', 'education', 'experience', 'seniority']].values.astype(np.float32)
valueY = dataset[['salary']].values.astype(np.float32)

#Separe the train and test
trainX, testX, trainY, testY = train_test_split(valueX, valueY, test_size=0.4, random_state=42)

#Converting to tensors
tensor_trainX = torch.tensor(trainX)
tensor_trainY = torch.tensor(trainY)
tensor_testX = torch.tensor(testX)
tensor_testY = torch.tensor(testY)

#Defining the modelo Linear Regression
class LinearRegression(nn.Module):
    #Chamada do construtor da classe herdada
    def __init__(self):
        super(LinearRegression, self).__init__(),
        #Atribuição de valores para o modelo que será utilizado
        self.linear = nn.Linear(4, 1)
    
    #Método responsável por chamar o método principal
    def forward(self, x):
        return self.linear(x)
    
modelLR = LinearRegression()

#Createing the loss and optimizer function
FuncMSE = nn.MSELoss() #Cálculo da perda
FuncOptimizer = torch.optim.SGD(modelLR.parameters(), lr=0.00001) #Função que atualiza os pesos

#Training
varEpochs = 1000

for epoch in range(varEpochs):
    modelLR.train()
    funcPred = modelLR(tensor_trainX)
    funcLoss = FuncMSE(funcPred, tensor_trainY)

    FuncOptimizer.zero_grad()
    funcLoss.backward()
    FuncOptimizer.step()

    if(epoch + 1) % 100 == 0:
        print(f"Período {epoch+1}/{varEpochs} -- Loss: {funcLoss.item():.4f}")


#Evaluate
modelLR.eval()
with torch.no_grad():
    funcPred_test = modelLR(tensor_testX)
    print("\nPrevisão: ")
    for i in range(len(testX)):
        print(f"Entrada: {testX[i]} -- Real: {testY[i][0]} | Previsto: {funcPred_test[i].item():.2f}")

#Show the bias and weights
for nome, parametro in modelLR.named_parameters():
    print(f"{nome}: {parametro.data.numpy()}")
    