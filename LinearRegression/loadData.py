import pandas as pd

class LoadData():

    def __init__(self):
        self.dataset = pd.read_csv('/home/ryan/Documents/Python/AI/AI_Training_Models/LinearRegression/sample_employees.csv')

    def showDataSet(self):
        print(self.dataset)

    def getData(self):
        return self.dataset

A = LoadData()
A.showDataSet()