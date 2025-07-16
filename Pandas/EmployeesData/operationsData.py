import pandas as pd
import numpy as np
from loadData import LoadData

class OperationsData():
    def __init__(self):
        self.dataL = LoadData()
        self.getData = self.dataL.get_data()

    def getEmployeeName(self, name):
        print(f"Resultado da busca:\n{self.getData[self.getData['name'] == name]}")

    def updateEmployeeName(self, name: str, column: str, value: str | int):
        self.getData.loc[self.getData['name'] == name, column] = value
        self.getData.to_csv('/home/ryan/Documents/Python/AI/AI_Training_Models/LinearRegression/Pandas/EmployeesData/employee.csv', index=False)
        print(f"Dados alterados:\n{self.getData[self.getData['name'] == name]}")
    
    def addNewColumn(self, column):
        self.getData[column] = "NULL"
        self.getData.to_csv('/home/ryan/Documents/Python/AI/AI_Training_Models/LinearRegression/Pandas/EmployeesData/employee.csv', index=False)
        print(f"Dados alterados:\n{self.getData.to_string()}")

    def deleteRow(self, name):
        self.getData = self.getData.loc[self.getData['name'] != name]
        self.getData.to_csv('/home/ryan/Documents/Python/AI/AI_Training_Models/LinearRegression/Pandas/EmployeesData/employee.csv', index=False)
        print(f"Dados alterados:\n{self.getData.to_string()}")

    def deleteData(self, name, column):
        self.getData.loc[self.getData['name'] == name, column] = np.nan
        self.getData.to_csv('/home/ryan/Documents/Python/AI/AI_Training_Models/LinearRegression/Pandas/EmployeesData/employee.csv', index=False)
        print(f"Dados alterados:\n{self.getData[self.getData['name'] == name]}")
        