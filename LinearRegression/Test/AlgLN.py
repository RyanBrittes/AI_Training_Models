import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('/home/ryan/Documents/Python/AI/AI_Training_Models/LinearRegression/sample_employees.csv')

yTrue = dataset['salary'].values
x = dataset[['age', 'education', 'experience', 'seniority']].values

w = np.zeros(x.shape[1])
b = 0
lr = 0.0004
epochs = 2000
n = len(yTrue)

losses = []

def funcMSE(yTrue, yPred):
    return np.mean((yTrue - yPred) ** 2)


for i in range(epochs):
    yPred = x @ w + b
    error = yPred - yTrue

    dw = (2/n) * (x.T @ error)
    db = (2/n) * np.sum(error)

    w -= dw * lr
    b -= db * lr

    lossValue = (funcMSE(yTrue, yPred))
    losses.append(lossValue)

    if ((i + 1) % 200) == 0:
        print(f"Loss --> {lossValue}")

