import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('/home/ryan/Documents/Python/AI/AI_Training_Models/files/diabetes.csv')

xValue = dataset[['Pregnancies']].values
yValue = dataset[['Age']].values

def meanFunction(rawValue):
    sumValue = 0
    for i in range(len(rawValue)):
        sumValue += rawValue[i]

    return sumValue/len(rawValue)

def standardDeviationFunction(rawValue):
    mean = meanFunction(rawValue)
    sumValue = 0
    for i in range(len(rawValue)):
        sumValue += (rawValue[i] - mean) ** 2
    
    return (sumValue / len(rawValue)) ** 0.5

def scoreZFunction(rawValue):
    meanValue = meanFunction(rawValue)
    stdValue = standardDeviationFunction(rawValue)
    listValue = []

    for i in range(len(rawValue)):
        value = (rawValue[i] - meanValue) / stdValue
        listValue.append(value)
    
    return np.vstack(listValue)


