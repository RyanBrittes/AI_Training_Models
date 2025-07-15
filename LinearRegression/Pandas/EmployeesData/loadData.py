import pandas as pd

class LoadData():

    def __init__(self):
        self.data = pd.read_csv('/home/ryan/Documents/Python/AI/AI_Training_Models/LinearRegression/Pandas/EmployeesData/employee.csv')

    def show_data(self):
        print(self.data)

    def get_data(self):
        return self.data
    