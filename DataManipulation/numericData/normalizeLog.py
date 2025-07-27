import pandas as pd
import numpy as np

dataset = pd.read_csv('/home/ryan/Documents/Python/AI/AI_Training_Models/files/diabetes.csv')

xValue = dataset[['Pregnancies']].values
yValue = dataset[['Age']].values

def logNormalizeFunction(rawValue):
    listValue = []
    const_not_zero = 1e-8
    for i in range(len(rawValue)):
        listValue.append(np.log(rawValue[i] + const_not_zero))
    print(np.vstack(listValue))

logNormalizeFunction(xValue)